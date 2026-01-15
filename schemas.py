from typing import Optional
from pydantic import Field, BaseModel
import yaml

class RerankerCollatorConfig(BaseModel):
    num_docs: int            
    num_negs: int

    doc_max_tokens: int        
    max_length: int

    pad_to_multiple_of: int
    add_query_emb_at_start: bool
    return_text: bool

class ModelConfig(BaseModel):
    backbone_name_or_path: str

    projector_in_dim: int
    projector_hidden_dim: int
    projector_out_dim: int
    trust_remote_code: bool

class DatasetsPath(BaseModel):
    train: str
    test: str

class LoraConfig(BaseModel):
    r: int
    alpha: int
    dropout: int

class TrainConfig(BaseModel):
    datasets: DatasetsPath
    out_dir: str = "ckpt_stage1"

    lr: float
    weight_decay: float
    epochs: int 
    warmup_ratio: float 
    grad_clip: float

    micro_batch_size: int         
    global_batch_size: int     
    num_workers: int 

    temperature: float         
    enable_similar: bool

    accelerator: str = "cpu"
    devices: int
    precision: str

    log_every: int
    save_every: int

    collator: RerankerCollatorConfig
    model: ModelConfig
    lora: LoraConfig


def load_config(path: str) -> TrainConfig:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TrainConfig(**config_dict)
