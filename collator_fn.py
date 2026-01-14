from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase


SYSTEM_PROMPT = (
    "You are a search relevance expert who can determine\n"
    "a ranking of passages based on their relevance to the query."
)

USER_HEADER = (
    "I will provide you with k passages, each indicated by a numerical identifier.\n"
    "Rank the passages based on their relevance to query: "
)


@dataclass
class RerankerCollatorConfig:
    num_docs: int = 16               
    doc_max_tokens: int = 768          
    max_length: Optional[int] = None   
    pad_to_multiple_of: Optional[int] = 8

    add_query_emb_at_start: bool = True   
    add_generation_prompt: bool = False

    shuffle_docs: bool = False            
    return_text: bool = False            


class RerankerPromptCollator:
    """
    Builds Qwen3-style instruction prompt + markers <|doc_emb|>, <|query_emb|>.
    Produces tokenized batch with guaranteed marker positions.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, cfg: RerankerCollatorConfig):
        self.tok = tokenizer
        self.cfg = cfg

        self._ensure_special_tokens(["<|doc_emb|>", "<|query_emb|>"])

        self.doc_emb_id = self._single_token_id("<|doc_emb|>")
        self.query_emb_id = self._single_token_id("<|query_emb|>")

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _ensure_special_tokens(self, toks: List[str]) -> None:
        existing = set(self.tok.get_vocab().keys())
        missing = [t for t in toks if t not in existing]
        if missing:
            self.tok.add_special_tokens({"additional_special_tokens": missing})

    def _single_token_id(self, token: str) -> int:
        tid = self.tok.convert_tokens_to_ids(token)
        if tid is None or tid == self.tok.unk_token_id:
            raise ValueError(f"Token {token} is not in tokenizer vocab (or mapped to unk).")
        enc = self.tok(token, add_special_tokens=False).input_ids
        if len(enc) != 1:
            raise ValueError(
                f"Token {token} is not a single token for this tokenizer. "
                f"Got ids={enc}. Add it as special token properly."
            )
        return tid

    def _encode(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def _encode_trunc(self, text: str, max_tokens: int) -> List[int]:
        ids = self._encode(text)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        return ids

    def _build_one(
        self,
        query: str,
        docs: Sequence[str],
        add_query_emb_at_start: bool,
    ) -> List[int]:
        """
        Builds one sequence of token ids following Table 1.
        """
        k = len(docs)

        ids: List[int] = []
        ids += self._encode("<|im_start|>system\n")
        ids += self._encode(SYSTEM_PROMPT)
        ids += self._encode("\n<|im_end|>\n")

        ids += self._encode("<|im_start|>user\n")
        ids += self._encode(USER_HEADER)
        ids += self._encode(query)

        if add_query_emb_at_start:
            ids.append(self.query_emb_id)

        ids += self._encode("\n")

        for i, doc in enumerate(docs, start=1):
            ids += self._encode(f'<passage id="{i}">\n')
            ids += self._encode_trunc(doc, self.cfg.doc_max_tokens)
            ids.append(self.doc_emb_id)  
            ids += self._encode("\n</passage>\n")

        ids += self._encode("<query>\n")
        ids += self._encode(query)
        ids.append(self.query_emb_id)
        ids += self._encode("\n</query>\n<|im_end|>\n")

        if self.cfg.add_generation_prompt:
            ids += self._encode("<|im_start|>assistant\n")

        return ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_ids: List[List[int]] = []
        batch_aug_ids: Optional[List[List[int]]] = [] if self._has_aug(features) else None

        pos_index: List[int] = []
        raw_text: Optional[List[str]] = [] if self.cfg.return_text else None
        raw_text_aug: Optional[List[str]] = [] if (self.cfg.return_text and batch_aug_ids is not None) else None

        for ex in features:
            query = ex["query"]
            pos = ex["pos"]
            neg = list(ex.get("neg", []))

            docs = [pos] + neg
            if len(docs) < self.cfg.num_docs:
                docs = docs + [""] * (self.cfg.num_docs - len(docs))
            else:
                docs = docs[: self.cfg.num_docs]

            if self.cfg.shuffle_docs:
                import random
                pairs = list(enumerate(docs))
                random.shuffle(pairs)
                docs = [d for _, d in pairs]
                pidx = [j for j, (orig_i, _) in enumerate(pairs) if orig_i == 0][0]
            else:
                pidx = 0

            ids = self._build_one(
                query=query,
                docs=docs,
                add_query_emb_at_start=self.cfg.add_query_emb_at_start,
            )

            if self.cfg.max_length is not None and len(ids) > self.cfg.max_length:
                raise RuntimeError(
                    f"Sequence length {len(ids)} exceeds max_length={self.cfg.max_length}. "
                    f"Reduce doc_max_tokens/num_docs or increase max_length."
                )

            batch_ids.append(ids)
            pos_index.append(pidx)

            if raw_text is not None:
                raw_text.append(self.tok.decode(ids, skip_special_tokens=False))

            if batch_aug_ids is not None:
                pos_aug = ex.get("pos_aug", pos)
                neg_aug = list(ex.get("neg_aug", neg))
                docs_aug = [pos_aug] + neg_aug
                if len(docs_aug) < self.cfg.num_docs:
                    docs_aug = docs_aug + [""] * (self.cfg.num_docs - len(docs_aug))
                else:
                    docs_aug = docs_aug[: self.cfg.num_docs]

                if self.cfg.shuffle_docs:
                    raise RuntimeError("shuffle_docs=True with docs_aug is not supported in this minimal collator.")

                ids_aug = self._build_one(
                    query=query,
                    docs=docs_aug,
                    add_query_emb_at_start=self.cfg.add_query_emb_at_start,
                )

                if self.cfg.max_length is not None and len(ids_aug) > self.cfg.max_length:
                    raise RuntimeError(
                        f"Aug sequence length {len(ids_aug)} exceeds max_length={self.cfg.max_length}. "
                        f"Reduce doc_max_tokens/num_docs or increase max_length."
                    )

                batch_aug_ids.append(ids_aug)
                if raw_text_aug is not None:
                    raw_text_aug.append(self.tok.decode(ids_aug, skip_special_tokens=False))

        out = self._pad(batch_ids)
        out["pos_index"] = torch.tensor(pos_index, dtype=torch.long)


        if batch_aug_ids is not None:
            out_aug = self._pad(batch_aug_ids)
            out["input_ids_aug"] = out_aug["input_ids"]
            out["attention_mask_aug"] = out_aug["attention_mask"]

        if raw_text is not None:
            out["text"] = raw_text 
        if raw_text_aug is not None:
            out["text_aug"] = raw_text_aug

        return out

    def _has_aug(self, features: List[Dict[str, Any]]) -> bool:
        return any(("pos_aug" in ex) or ("neg_aug" in ex) for ex in features)

    def _pad(self, batch_ids: List[List[int]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x) for x in batch_ids)
        if self.cfg.pad_to_multiple_of:
            m = self.cfg.pad_to_multiple_of
            if max_len % m != 0:
                max_len = ((max_len // m) + 1) * m

        pad_id = self.tok.pad_token_id
        input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch_ids), max_len), dtype=torch.long)

        for i, ids in enumerate(batch_ids):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attn[i, :L] = 1

        return {"input_ids": input_ids, "attention_mask": attn}
