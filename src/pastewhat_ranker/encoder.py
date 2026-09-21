"""Trainable ModernBERT encoder, adapted from laya-coreml (Apache-2.0).

Parameter names exactly match the non-quantized Laya checkpoint. Original typed
decision heads are deliberately absent. See provenance/initialization.json.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg.get("norm_eps", 1e-5),
                                 bias=cfg.get("norm_bias", False))

    def forward(self, ids):
        return self.norm(self.tok_embeddings(ids))


class Attention(nn.Module):
    def __init__(self, cfg, kind, max_length):
        super().__init__()
        d = cfg["hidden_size"]
        self.heads, self.dim = cfg["num_attention_heads"], d // cfg["num_attention_heads"]
        self.Wqkv = nn.Linear(d, 3 * d, bias=cfg.get("attention_bias", False))
        self.Wo = nn.Linear(d, d, bias=cfg.get("attention_bias", False))
        params = cfg.get("rope_parameters", {}).get(kind, {})
        if params.get("rope_type", "default") != "default":
            raise ValueError("Only default RoPE is supported")
        fallback = cfg.get("global_rope_theta", 160000.0) if kind == "full_attention" else cfg.get("local_rope_theta", 10000.0)
        base = params.get("rope_theta", fallback)
        inv = 1.0 / (float(base) ** (torch.arange(0, self.dim, 2).float() / self.dim))
        angles = torch.outer(torch.arange(max_length).float(), inv)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rotate(self, value):
        a, b = value.chunk(2, dim=-1)
        length = value.shape[-2]
        c = self.cos[:, :, :length].to(value.dtype)
        s = self.sin[:, :, :length].to(value.dtype)
        return torch.cat((a * c - b * s, b * c + a * s), dim=-1)

    def forward(self, x, mask):
        b, n, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, n, 3, self.heads, self.dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        value = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, attn_mask=mask)
        return self.Wo(value.transpose(1, 2).reshape(b, n, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["hidden_size"], cfg["intermediate_size"]
        self.Wi = nn.Linear(d, 2 * h, bias=cfg.get("mlp_bias", False))
        self.Wo = nn.Linear(h, d, bias=cfg.get("mlp_bias", False))

    def forward(self, x):
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(value) * gate)


class Layer(nn.Module):
    def __init__(self, cfg, index, kind, max_length):
        super().__init__()
        self.kind = kind

        def norm():
            return nn.LayerNorm(cfg["hidden_size"], eps=cfg.get("norm_eps", 1e-5), bias=cfg.get("norm_bias", False))

        self.attn_norm = nn.Identity() if index == 0 else norm()
        self.attn = Attention(cfg, kind, max_length)
        self.mlp_norm = norm()
        self.mlp = MLP(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class Encoder(nn.Module):
    def __init__(self, cfg, max_length=1024):
        super().__init__()
        if cfg.get("model_type") != "modernbert" or cfg.get("hidden_activation", "gelu") != "gelu":
            raise ValueError("Only GELU ModernBERT checkpoints are supported")
        self.embeddings = Embeddings(cfg)
        kinds = cfg.get("layer_types") or ["full_attention" if i % cfg.get("global_attn_every_n_layers", 3) == 0
                                            else "sliding_attention" for i in range(cfg["num_hidden_layers"])]
        if len(kinds) != cfg["num_hidden_layers"] or set(kinds) - {"full_attention", "sliding_attention"}:
            raise ValueError("Invalid layer_types")
        self.layers = nn.ModuleList([Layer(cfg, i, kind, max_length) for i, kind in enumerate(kinds)])
        self.final_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg.get("norm_eps", 1e-5), bias=cfg.get("norm_bias", False))
        self.window = cfg.get("local_attention", 128)
        self.gradient_checkpointing = False
        self.register_buffer("positions", torch.arange(max_length, dtype=torch.int32), persistent=False)

    def forward(self, ids, valid):
        x = self.embeddings(ids)
        full = valid[:, None, None, :]
        positions = self.positions[:ids.shape[1]]
        window = (positions[:, None] - positions[None, :]).abs() <= self.window // 2
        local = (window[None, None] | ~valid[:, None, :, None]) & full
        for layer in self.layers:
            mask = full if layer.kind == "full_attention" else local
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        return self.final_norm(x)
