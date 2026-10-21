"""The transformer of Shai et al. (2024), Appendix A.6. By default: 4 layers, d_model 64, one attention head of
dimension 8, ReLU MLPs of width 256, pre-LayerNorm, learned positional embeddings, causal attention, no dropout."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab: int = 2
    context: int = 32
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 1
    d_head: int = 8
    d_mlp: int = 256


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.n_heads * cfg.d_head)
        self.out = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.ReLU(), nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the updated residual stream and the normalized block input (a probe feature)."""
        normalized = self.ln1(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(normalized).view(batch, length, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(attention.transpose(1, 2).reshape(batch, length, -1))
        x = x + self.mlp(self.ln2(x))
        return x, normalized


class Transformer(nn.Module):
    def __init__(self, cfg: Config, init_std: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab)
        for module in self.modules():  # every weight matrix ~ N(0, init_std^2); biases 0; LayerNorm 1 and 0
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=init_std)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, features: bool = False):
        """Logits (batch, length, vocab). With features=True, also the probe features
        (batch, length, (n_layers + 1) d_model): the residual stream at every layer as the network reads it, i.e.
        the LayerNorm-ed input of each block and the LayerNorm-ed output of the last block (the input to the
        unembedding)."""
        x = self.embed(tokens) + self.pos_embed(torch.arange(tokens.shape[1], device=tokens.device))
        normalized = []
        for block in self.blocks:
            x, block_input = block(x)
            normalized.append(block_input)
        final = self.ln_final(x)
        logits = self.unembed(final)
        return (logits, torch.cat(normalized + [final], -1)) if features else logits


def untrained(cfg: Config, init_std: float, seed: int) -> Transformer:
    """The initial network of the run with this seed. train.py starts from it, and probe.py regenerates it from the
    seed rather than storing it."""
    torch.manual_seed(seed)
    return Transformer(cfg, init_std)
