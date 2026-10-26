"""The transformer of Shai et al. (2024), Appendix A.6. By default: 4 layers, d_model 64, one attention head of
dimension 8, ReLU MLPs of width 256, pre-LayerNorm, learned positional embeddings, causal attention, no dropout.
With rope=True, rotary position embeddings (Su et al. 2021) on the queries and keys replace the learned positional
embeddings. The network reads a beginning-of-sequence (BOS) token, an extra row of the embedding, followed by the
tokens, so that its outputs predict every token of a sequence, including the first. Only position 0 attends to BOS:
the first prediction is the network's output on BOS alone, and every later position reads only the tokens, as it would
without BOS. BOS thus gives no later position a marker of its distance from the start."""
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
    rope: bool = False


def rotary(x: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """Rotary position embedding of queries or keys (batch, heads, length, d_head): the two halves of each vector are
    rotated as pairs of coordinates by angles position * base^(-i / (d_head / 2)), i = 0, ..., d_head / 2 - 1."""
    half = x.shape[-1] // 2
    frequencies = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * frequencies
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads, self.d_head, self.rope = cfg.n_heads, cfg.d_head, cfg.rope
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.n_heads * cfg.d_head)
        self.out = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.ReLU(), nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the updated residual stream and the normalized block input (a probe feature). mask (length,
        length) is True where a query position may attend to a key position."""
        normalized = self.ln1(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(normalized).view(batch, length, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotary(q), rotary(k)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(attention.transpose(1, 2).reshape(batch, length, -1))
        x = x + self.mlp(self.ln2(x))
        return x, normalized


class Transformer(nn.Module):
    def __init__(self, cfg: Config, init_std: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab + 1, cfg.d_model)  # token cfg.vocab is BOS
        self.pos_embed = None if cfg.rope else nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab)
        for module in self.modules():  # every weight matrix ~ N(0, init_std^2); biases 0; LayerNorm 1 and 0
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=init_std)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, features: bool = False):
        """Logits (batch, length, vocab): position n predicts tokens[:, n] from tokens[:, :n]. The network reads BOS
        followed by tokens[:, :-1], so position n has seen n tokens and the last token is only a target. Attention
        is causal, and only position 0 attends to BOS. With
        features=True, also the probe features (batch, length, (n_layers + 1) d_model): the residual stream at every
        layer as the network reads it, i.e. the LayerNorm-ed input of each block and the LayerNorm-ed output of the
        last block (the input to the unembedding)."""
        inputs = torch.cat([torch.full_like(tokens[:, :1], self.cfg.vocab), tokens[:, :-1]], 1)
        x = self.embed(inputs)
        if self.pos_embed is not None:
            x = x + self.pos_embed(torch.arange(tokens.shape[1], device=tokens.device))
        length = tokens.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
        mask[1:, 0] = False  # BOS is seen only by itself
        normalized = []
        for block in self.blocks:
            x, block_input = block(x, mask)
            normalized.append(block_input)
        final = self.ln_final(x)
        logits = self.unembed(final)
        return (logits, torch.cat(normalized + [final], -1)) if features else logits


def untrained(cfg: Config, init_std: float, seed: int) -> Transformer:
    """The initial network of the run with this seed. train.py starts from it, and probe.py regenerates it from the
    seed rather than storing it."""
    torch.manual_seed(seed)
    return Transformer(cfg, init_std)
