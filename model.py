"""The transformer of Shai et al. (2024), Appendix A.6. By default: 4 layers, d_model 64, one attention head of
dimension 8, ReLU MLPs of width 256, pre-LayerNorm, causal attention, no dropout.
By default the queries and keys use rotary position embeddings (Su et al. 2021). The learned absolute embedding
comparison uses rope=False. The network reads a beginning-of-sequence (BOS) token, an extra row of the embedding, followed by the
tokens, so that its outputs predict every token of a sequence, including the first."""
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
    rope: bool = True
    bos_visible: bool = True


def rotary(x: torch.Tensor, base: float = 10_000.0, *, positions=None) -> torch.Tensor:
    """Rotary position embedding of queries or keys (batch, heads, length, d_head): the two halves of each vector are
    rotated as pairs of coordinates by angles position * base^(-i / (d_head / 2)), i = 0, ..., d_head / 2 - 1."""
    half = x.shape[-1] // 2
    frequencies = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    if positions is None:
        positions = torch.arange(x.shape[-2], device=x.device)
    angles = torch.as_tensor(positions, device=x.device)[..., None] * frequencies
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

    def forward(self, x: torch.Tensor, mask=None, *, positions=None, cache=None):
        """Full or cached execution, returning residuals, probe inputs and (K, V).

        Cache tensors have shape (batch, heads, positions, d_head). Keys already
        include RoPE. Cache length determines positions unless supplied explicitly;
        gradients through written keys and values are preserved.
        """
        normalized = self.ln1(x)
        batch, length, _ = x.shape
        past = 0 if cache is None else cache[0].shape[-2]
        if positions is None:
            positions = torch.arange(past, past+length, device=x.device)
        q, k, v = self.qkv(normalized).view(batch, length, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotary(q, positions=positions), rotary(k, positions=positions)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        if mask is None and past:
            mask = torch.arange(past+length, device=x.device)[None] <= past+torch.arange(length, device=x.device)[:, None]
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        x = x+self.out(attention.transpose(1, 2).reshape(batch, length, -1))
        return x+self.mlp(self.ln2(x)), normalized, (k, v)


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
        followed by tokens[:, :-1], so position n has seen n tokens and the last token is only a target. With
        features=True, also the probe features (batch, length, (n_layers + 1) d_model): the residual stream at every
        layer as the network reads it, i.e. the LayerNorm-ed input of each block and the LayerNorm-ed output of the
        last block (the input to the unembedding)."""
        inputs = torch.cat([torch.full_like(tokens[:, :1], self.cfg.vocab), tokens[:, :-1]], 1)
        x = self.embed(inputs)
        if self.pos_embed is not None:
            x = x + self.pos_embed(torch.arange(tokens.shape[1], device=tokens.device))
        mask = None
        if not self.cfg.bos_visible:
            mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device).tril()
            mask[1:, 0] = False
        normalized = []
        for block in self.blocks:
            x, block_input, _ = block(x, mask)
            normalized.append(block_input)
        final = self.ln_final(x)
        logits = self.unembed(final)
        return (logits, torch.cat(normalized + [final], -1)) if features else logits


def untrained(cfg: Config, init_std: float, seed: int) -> Transformer:
    """The initial network of the run with this seed. train.py starts from it, and probe.py regenerates it from the
    seed rather than storing it."""
    torch.manual_seed(seed)
    return Transformer(cfg, init_std)
