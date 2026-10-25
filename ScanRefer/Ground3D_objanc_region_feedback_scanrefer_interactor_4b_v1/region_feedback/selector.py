"""Ordinary MLP selector control: no second-pass Qwen computation."""
import torch
from torch import nn

class OrdinarySelector(nn.Module):
    def __init__(self,input_dim,language_dim,hidden_dim,num_queries):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(input_dim+2*language_dim,hidden_dim),
                                   nn.GELU(),nn.Linear(hidden_dim,num_queries))
    def forward(self,records):
        rows=[];device=self.network[0].weight.device;dtype=self.network[0].weight.dtype
        for record in records:
            evidence=record["evidence"]
            features=evidence["features"][evidence["valid_mask"]]
            objects=features.mean(0) if len(features) else evidence["features"].new_zeros(evidence["features"].shape[-1])
            row=torch.cat([record["scene"].float().mean(0),record["question_mean"].float(),objects.float()])
            rows.append(row.to(device=device,dtype=dtype))
        return self.network(torch.stack(rows))
