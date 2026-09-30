"""GPT-like Transformer with hand-written activation checkpointing (no torch.utils.checkpoint)
and optional offloading of saved activations to host RAM."""
import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_fn, preserve_rng, x):
        ctx.run_fn = run_fn
        ctx.preserve_rng = preserve_rng
        if preserve_rng:  # recompute must see the same dropout masks
            ctx.cpu_rng = torch.get_rng_state()
            ctx.cuda_rng = torch.cuda.get_rng_state(x.device) if x.is_cuda else None

        device_type = x.device.type
        ctx.autocast = dict(
            device_type=device_type,
            enabled=torch.is_autocast_enabled(device_type),
            dtype=torch.get_autocast_dtype(device_type),
            cache_enabled=torch.is_autocast_cache_enabled(),
        )
        ctx.save_for_backward(x)
        with torch.no_grad():
            return run_fn(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        x = x.detach().requires_grad_(True)

        rng_devices = [x.device] if ctx.preserve_rng and x.is_cuda else []
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng):
            if ctx.preserve_rng:
                torch.set_rng_state(ctx.cpu_rng)
                if ctx.cuda_rng is not None:
                    torch.cuda.set_rng_state(ctx.cuda_rng, x.device)
            with record_function("ckpt_recompute"), torch.enable_grad(), torch.autocast(**ctx.autocast):
                out = ctx.run_fn(x)

        torch.autograd.backward(out, grad_out)
        return None, None, x.grad


def checkpoint(fn, x, preserve_rng=True):
    return CheckpointFunction.apply(fn, preserve_rng, x)


@dataclass
class Config:
    vocab_size: int = 8192
    d_model: int = 1024
    n_head: int = 16
    n_layer: int = 12
    max_seq_len: int = 32768
    dropout: float = 0.0
    use_checkpoint: bool = False
    use_offload: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        self.idx = idx
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        with record_function(f"block_{self.idx}"):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_checkpoint = cfg.use_checkpoint
        self.use_offload = cfg.use_offload
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        ckpt = self.use_checkpoint and self.training and torch.is_grad_enabled()
        offload = self.use_offload and self.training and torch.is_grad_enabled()
        # Saved activations of the blocks go to pinned host RAM in forward and come back in backward.
        ctx = torch.autograd.graph.save_on_cpu(pin_memory=x.is_cuda) if offload else contextlib.nullcontext()
        with ctx:
            for block in self.blocks:
                x = checkpoint(block, x) if ckpt else block(x)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
