from dataclasses import dataclass 
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel
from typing import List, Optional, Tuple


def extract_marker_embeddings(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    marker_id: int,
    max_markers: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract embeddings at positions where input_ids == marker_id.

    hidden:    (B, L, H)
    input_ids: (B, L)

    Returns:
      embs:   (B, M, H) padded with zeros, where M = max markers in batch (or max_markers)
      counts: (B,) number of markers for each sample
    """
    B, L, H = hidden.shape
    mask = input_ids.eq(marker_id)                # (B, L)
    counts = mask.sum(dim=1)                      # (B,)

    if max_markers is None:
        M = int(counts.max().item()) if B > 0 else 0
    else:
        M = int(max_markers)

    embs = hidden.new_zeros((B, M, H))

    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False).squeeze(1) 
        if idx.numel() == 0:
            continue
        if M == 0:
            continue
        if idx.numel() > M:
            idx = idx[:M]
        embs[b, : idx.numel(), :] = hidden[b, idx, :]

    return embs, counts


def pick_first_and_last(
    embs: torch.Tensor,
    counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    embs:   (B, M, H) marker embeddings padded
    counts: (B,) number of markers per sample

    Returns:
      first: (B, H)
      last:  (B, H)  (if count==0 -> zeros)
    """
    B, M, H = embs.shape
    first = embs[:, 0, :] if M > 0 else embs.new_zeros((B, H))

    last = embs.new_zeros((B, H))
    for b in range(B):
        c = int(counts[b].item())
        if c <= 0:
            continue
        last[b] = embs[b, c - 1]
    return first, last


@dataclass
class RerankerForwardOutput:
    q_end: torch.Tensor
    q_start: Optional[torch.Tensor]
    docs: torch.Tensor
    scores: torch.Tensor


class MLPProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class JinaReranker(nn.Module):
    def __init__(
            self,
            backbone_name_or_path: str,
            tokenizer,
            doc_emb_token_id: int,
            query_emb_token_id: int,
            projector_in_dim: int = 1024,
            projector_hidden_dim: int = 512,
            projector_out_dim: int = 512,
            trust_remote_code: bool = True
        ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name_or_path, trust_remote_code=trust_remote_code, output_hidden_states=True)
        self.backbone.resize_token_embeddings(len(tokenizer))

        self.doc_emb_token_id = int(doc_emb_token_id)
        self.query_emb_token_id = int(query_emb_token_id)
        self.projector = MLPProjection(projector_in_dim, projector_hidden_dim, projector_out_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_docs: Optional[int] = None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state

        doc_ctx, doc_counts = extract_marker_embeddings(
            hidden, input_ids, self.doc_emb_token_id, max_markers=max_docs
        )
        
        print(doc_counts)

        q_ctx, q_counts = extract_marker_embeddings(hidden, input_ids, self.query_emb_token_id)
        q_first_ctx, q_last_ctx = pick_first_and_last(q_ctx, q_counts)

        docs = self.projector(doc_ctx)
        q_end = self.projector(q_last_ctx)

        has_start = (q_counts >= 2).all().item()

        q_start = self.projector(q_first_ctx) if has_start else None

        docs_n = F.normalize(docs)
        q_end_n = F.normalize(q_end)
        scores = (docs_n * q_end_n.unsqueeze(1)).sum(dim=-1)  # (B, K)

        return RerankerForwardOutput(q_end=q_end, q_start=q_start, docs=docs, scores=scores)

    

