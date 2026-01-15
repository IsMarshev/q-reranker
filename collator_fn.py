import torch
from transformers import PreTrainedTokenizerBase

from typing import Any, Dict, List, Optional, Sequence
from schemas import RerankerCollatorConfig


SYSTEM_PROMPT = (
    "You are a search relevance expert who can determine\n"
    "a ranking of passages based on their relevance to the query."
)

USER_HEADER = (
    "I will provide you with k passages, each indicated by a numerical identifier.\n"
    "Rank the passages based on their relevance to query: "
)

class Stage1PromptCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, cfg: RerankerCollatorConfig):
        self.tok = tokenizer
        self.cfg = cfg

        self._ensure_special_tokens(["<|doc_emb|>", "<|query_emb|>"])
        self.doc_emb_id = self._single_token_id("<|doc_emb|>")
        self.query_emb_id = self._single_token_id("<|query_emb|>")

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        # pre-encode static parts for budgeting
        self._sys_prefix = self._enc("<|im_start|>system\n") + self._enc(SYSTEM_PROMPT) + self._enc("\n<|im_end|>\n")
        self._user_prefix_a = self._enc("<|im_start|>user\n") + self._enc(USER_HEADER)
        self._user_prefix_b = self._enc("\n")

        self._passage_open_tpl = '<passage id="{i}">\n'
        self._passage_close = self._enc("\n</passage>\n")

        self._query_open = self._enc("<query>\n")
        self._query_close = self._enc("\n</query>\n<|im_end|>\n")

        # Estimate per-passage overhead (tags + marker + newlines) in tokens
        # (doc text budget will be computed to fit max_length).
        open_ids = self._enc(self._passage_open_tpl.format(i=1))
        self._per_doc_overhead = len(open_ids) + 1 + len(self._passage_close)  # +1 for <|doc_emb|>

    def _ensure_special_tokens(self, toks: List[str]) -> None:
        vocab = set(self.tok.get_vocab().keys())
        missing = [t for t in toks if t not in vocab]
        if missing:
            self.tok.add_special_tokens({"additional_special_tokens": missing})

    def _single_token_id(self, token: str) -> int:
        tid = self.tok.convert_tokens_to_ids(token)
        if tid is None or tid == self.tok.unk_token_id:
            raise ValueError(f"Token {token} missing in tokenizer vocab.")
        enc = self.tok(token, add_special_tokens=False).input_ids
        if len(enc) != 1:
            raise ValueError(f"Token {token} is not a single token: {enc}")
        return tid

    def _enc(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_ids: List[List[int]] = []
        batch_text: Optional[List[str]] = [] if self.cfg.return_text else None

        for ex in features:
            q = ex["query"]
            pos = ex["pos"]
            negs = list(ex.get("neg", [])) or list(ex.get("negs", []))

            if len(negs) < self.cfg.num_negs:
                raise ValueError(f"Need at least {self.cfg.num_negs} negatives, got {len(negs)}")

            # sample 15 negs if more
            if len(negs) > self.cfg.num_negs:
                negs = random.sample(negs, self.cfg.num_negs)

            docs = [pos] + negs
            assert len(docs) == self.cfg.num_docs

            ids = self._build_one(q, docs)

            batch_ids.append(ids)
            if batch_text is not None:
                batch_text.append(self.tok.decode(ids, skip_special_tokens=False))

        out = self._pad(batch_ids)
        out["pos_index"] = torch.zeros(len(features), dtype=torch.long)  # pos always at doc#1
        if batch_text is not None:
            out["text"] = batch_text
        return out

    def _doc_text_budget(self, query: str) -> int:
        """
        Choose doc text token budget so that full prompt fits max_length.
        We keep *all 16 docs* and always keep doc_emb markers.
        """
        # header lengths depend on query tokens
        q_ids = self._enc(query)
        header_len = len(self._sys_prefix) + len(self._user_prefix_a) + len(q_ids)
        if self.cfg.add_query_emb_at_start:
            header_len += 1  # <|query_emb|>
        header_len += len(self._user_prefix_b)

        footer_len = len(self._query_open) + len(q_ids) + 1 + len(self._query_close)  # +1 for <|query_emb|>

        static_len = header_len + footer_len + self.cfg.num_docs * self._per_doc_overhead
        remaining = self.cfg.max_length - static_len
        if remaining <= 0:
            # fallback: very small budget; still keep markers
            return 1
        per_doc = remaining // self.cfg.num_docs
        return max(1, min(self.cfg.doc_max_tokens, per_doc))

    def _build_one(self, query: str, docs: List[str]) -> List[int]:
        q_ids = self._enc(query)
        budget = self._doc_text_budget(query)

        ids: List[int] = []
        ids += self._sys_prefix
        ids += self._user_prefix_a
        ids += q_ids
        if self.cfg.add_query_emb_at_start:
            ids.append(self.query_emb_id)  # training-time extra marker :contentReference[oaicite:6]{index=6}
        ids += self._user_prefix_b

        for i, doc in enumerate(docs, start=1):
            ids += self._enc(self._passage_open_tpl.format(i=i))
            doc_ids = self._enc(doc)
            doc_ids = doc_ids[:budget]
            ids += doc_ids
            ids.append(self.doc_emb_id)
            ids += self._passage_close

        ids += self._query_open
        ids += q_ids
        ids.append(self.query_emb_id)
        ids += self._query_close

        # if still overflow (should be rare), hard-truncate from inside doc text (keep tail markers by cutting earlier)
        if len(ids) > self.cfg.max_length:
            ids = ids[: self.cfg.max_length]
        return ids

    def _pad(self, batch_ids: List[List[int]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x) for x in batch_ids)
        m = self.cfg.pad_to_multiple_of
        if m and (max_len % m != 0):
            max_len = ((max_len // m) + 1) * m

        pad = self.tok.pad_token_id
        input_ids = torch.full((len(batch_ids), max_len), pad, dtype=torch.long)
        attn = torch.zeros((len(batch_ids), max_len), dtype=torch.long)

        for i, ids in enumerate(batch_ids):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attn[i, :L] = 1

        return {"input_ids": input_ids, "attention_mask": attn}