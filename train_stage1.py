import os
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from logging import Logger

from dataclasses import dataclass

from collator_fn import RerankerCollatorConfig, Stage1PromptCollator
from model import JinaReranker
from loss import MultiLoss
from schemas import TrainConfig
from lightning.fabric import Fabric

from peft import LoraConfig, get_peft_model


class RerankerDataset(Dataset):
    """
    Lines
      {"query": "...", "pos": "...", "neg": ["...", "...", ...]}
    """
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.logger = Logger("Trainer stage 1")
        self.cfg = cfg

        self.fabric = Fabric(accelerator=self.cfg.accelerator, devices=self.cfg.devices, precision=self.cfg.precision)
        self.fabric.launch()
        
        model_config = self.cfg.model
        lora_config = self.cfg.lora
        self.tokenizer = self.init_tokenizer()
        self.logger.info("Токенайзер подготовлен")

        self.model = JinaReranker(
            backbone_name_or_path = model_config.backbone_name_or_path,
            tokenizer=self.tokenizer,
            doc_emb_token_id=self.tokenizer.convert_tokens_to_ids("<|doc_emb|>"),
            query_emb_token_id=self.tokenizer.convert_tokens_to_ids("<|query_emb|>"),
            projector_in_dim=model_config.projector_in_dim,
            projector_hidden_dim=model_config.projector_hidden_dim,
            projector_out_dim=model_config.projector_out_dim,
            trust_remote_code=model_config.trust_remote_code
            )
        self.logger.info("Модель загружена")
        self.model.backbone = self.apply_lora_qwen(r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        self.logger.info("ЛОРА конфиг приминен")
        self.emb = self.model.backbone.get_input_embeddings()
        if self.emb is not None:
            self.emb.weight.requires_grad = True

        self.criterion = MultiLoss(
                temperature=cfg.temperature,
                w_disperse=0.45,
                w_dual=0.85,
                w_similar=0.85,
                enable_similar=cfg.enable_similar,
        )

        self.train_data = RerankerDataset(self.cfg.datasets.train)
        collator = Stage1PromptCollator(self.tokenizer, cfg.collator)
        self.dl = DataLoader(
                            self.train_data,
                            batch_size=self.cfg.micro_batch_size, 
                            shuffle=True, 
                            num_workers=self.cfg.num_workers,
                            collate_fn=collator, 
                            pin_memory=True)
        self.optim = torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay
                )
        
        self.apply_fabric()
        
    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.backbone_name_or_path, trust_remote_code=self.cfg.model.trust_remote_code)
        new_special_tokens = {
            "additional_special_tokens": [
                "<|query_emb|>", 
                "<|doc_emb|>"
            ]
        }
        num_added_toks = tokenizer.add_special_tokens(new_special_tokens)
        self.logger.info(f"Добавлено новых токенов: {num_added_toks}")
        return tokenizer

    def apply_lora_qwen(self, r: int = 16, alpha: int = 32, dropout: float = 0.0) -> nn.Module:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        for p in self.model.backbone.parameters():
            p.requires_grad = False

        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_modules,
            task_type="FEATURE_EXTRACTION", 
        )
        self.model.backbone = get_peft_model(self.model.backbone, cfg)
        return self.model.backbone

    def apply_fabric(self):
        self.model, self.optim = self.fabric.setup(self.model, self.optim)
        self.dl = self.fabric.setup_dataloaders(self.dl)

    def train(self):
        world = self.fabric.world_size
        accum_steps = max(1, self.cfg.global_batch_size // (self.cfg.micro_batch_size * world))
        if self.fabric.is_global_zero:
            print(f"[fabric] world_size={world} micro_bs={self.cfg.micro_batch_size} accum_steps={accum_steps}")

        # Scheduler
        steps_per_epoch = math.ceil(len(self.dl) / accum_steps)
        total_steps = steps_per_epoch * self.cfg.epochs
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(self.optim, warmup_steps, total_steps)

        self.model.train()
        global_step = 0
        running = {}

        for epoch in range(self.cfg.epochs):
            for step, batch in enumerate(self.dl):
                is_accum = ((step + 1) % accum_steps) != 0

                with self.fabric.no_backward_sync(self.model, enabled=is_accum):
                    with self.fabric.autocast():
                        out = self.model(batch["input_ids"], batch["attention_mask"], max_docs=16)
                        loss_dict = self.criterion(q_end=out.q_end, q_start=out.q_start, docs=out.docs, docs_aug=None)
                        loss = loss_dict["loss"] / accum_steps

                    self.fabric.backward(loss)

                if not is_accum:
                    self.fabric.clip_gradients(self.model, self.optim, max_norm=self.cfg.grad_clip)
                    self.optim.step()
                    scheduler.step()
                    self.optim.zero_grad(set_to_none=True)

                    global_step += 1

                    if self.fabric.is_global_zero and (global_step % self.cfg.log_every == 0):
                        msg = (f"ep={epoch} step={global_step}/{total_steps} "
                            f"loss={loss_dict['loss'].item():.4f} "
                            f"rank={loss_dict['l_rank'].item():.4f} "
                            f"disp={loss_dict['l_disperse'].item():.4f} "
                            f"dual={loss_dict['l_dual'].item():.4f}")
                        if self.cfg.enable_similar:
                            msg += f" sim={loss_dict['l_similar'].item():.4f}"
                        print(msg)

                    if (global_step % self.cfg.save_every == 0) and self.fabric.is_global_zero:
                        save_dir = os.path.join(self.cfg.out_dir, f"step_{global_step}")
                        os.makedirs(save_dir, exist_ok=True)

                        torch.save(self.fabric.unwrap(self.model).projector.state_dict(), os.path.join(save_dir, "projector.pt"))

                        self.fabric.unwrap(self.model).backbone.save_pretrained(save_dir)
                        self.tokenizer.save_pretrained(save_dir)

                        print(f"[ckpt] saved to {save_dir}")

            if self.fabric.is_global_zero:
                save_dir = os.path.join(self.cfg.out_dir, f"epoch_{epoch}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.fabric.unwrap(self.model).projector.state_dict(), os.path.join(save_dir, "projector.pt"))
                self.fabric.unwrap(self.model).backbone.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                print(f"[ckpt] saved to {save_dir}")