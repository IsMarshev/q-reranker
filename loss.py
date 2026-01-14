import torch
import torch.nn as nn
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()  
        self.temperature = temperature

    def forward(self, 
                q: torch.Tensor,
                d_pos: torch.Tensor,
                d_neg: torch.Tensor) -> torch.Tensor:
        """
        ℓ_rank = -(1/N) Σ_i log exp(s(q_i, d_i+)/τ) / (exp(pos/τ) + Σ_k exp(neg_k/τ))

        q:     (N, D)
        d_pos: (N, D)
        d_neg: (N, K, D)
        """
        q = F.normalize(q, dim=-1)
        d_pos = F.normalize(d_pos, dim=-1)
        d_neg = F.normalize(d_neg, dim=-1)

        pos_sim = (q * d_pos).sum(dim=-1, keepdim=True)         
        neg_sim = (q.unsqueeze(1) * d_neg).sum(dim=-1)         

        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature    
        target = torch.zeros(q.size(0), dtype=torch.long, device=q.device)  

        return F.cross_entropy(logits, target)
    
class DisperseLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()  
        self.temperature = temperature

    def forward(self,
                d_pos: torch.Tensor,
                d_neg: torch.Tensor,
                eps: float = 1e-12) -> torch.Tensor:
        """
        ℓ_disperse = (1/N) Σ_i log [ (1/K) Σ_k ( exp(s(d_i+, d_i,k-)/τ) + Σ_{j=k+1..K} exp(s(d_i,k-, d_i,j-)/τ) ) ]

        d_pos: (N, D)
        d_neg: (N, K, D)
        """
        d_pos = F.normalize(d_pos, dim=-1)
        d_neg = F.normalize(d_neg, dim=-1)

        N, K, _ = d_neg.shape

        s_pos_neg = (d_pos.unsqueeze(1) * d_neg).sum(dim=-1)

        s_neg_neg = torch.matmul(d_neg, d_neg.transpose(1, 2))

        mask = torch.triu(torch.ones(K, K, device=d_neg.device, dtype=torch.bool), diagonal=1)
        exp_neg_neg = torch.exp(s_neg_neg / self.temperature).masked_fill(~mask, 0.0)  

        sum_future = exp_neg_neg.sum(dim=-1)

        term_k = torch.exp(s_pos_neg / self.temperature) + sum_future  
        mean_k = term_k.mean(dim=1)                      
        return torch.log(mean_k + eps).mean()
    

class SimilarityLoss(nn.Module):
    def __init__(self, temperature: float = 0.05, symmetric: bool = True):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(self, d: torch.Tensor, d_aug: torch.Tensor) -> torch.Tensor:
        """
        ℓ_similar: NT-Xent / SimCLR-style loss

        treats (d_i, d_i_aug) as positives, all other docs as negatives.
        d:     (N, D)  original docs
        d_aug: (N, D)  augmented docs (d*)
        """
        d = F.normalize(d, dim=-1)
        d_aug = F.normalize(d_aug, dim=-1)

        N = d.size(0)
        z = torch.cat([d, d_aug], dim=0) 

        logits = (z @ z.T) / self.temperature 

        logits.fill_diagonal_(-1e9)
        pos_index = torch.arange(2 * N, device=z.device)
        pos_index = (pos_index + N) % (2 * N)

        if self.symmetric:
            return F.cross_entropy(logits, pos_index)
        else:
            return F.cross_entropy(logits[:N], pos_index[:N])
