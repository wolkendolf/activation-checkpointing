"""Memory snapshot + time profile for baseline vs activation checkpointing vs RAM offloading.

    python bench.py --check              # grads match across baseline / checkpointing / offloading
    python bench.py --seq-len 16384      # -> results/{mode}_T{seq_len}.pickle / .json
    python bench.py --gpu 3              # run on physical GPU 3 only (default: 0)
"""
import argparse
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from model import Config, Transformer

RESULTS_DIR = "results"


def train_step(model, idx, targets):
    with record_function("## forward ##"), torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(idx, targets)[1]  # don't hold the T×V logits alive through backward
    with record_function("## backward ##"):
        loss.backward()
    model.zero_grad(set_to_none=True)


def profile_mode(mode, seq_len, steps):
    torch.manual_seed(0)
    model = Transformer(Config(max_seq_len=seq_len, use_checkpoint=mode == "ckpt", use_offload=mode == "offload")).cuda().train()
    idx, targets = torch.randint(model.cfg.vocab_size, (2, 1, seq_len), device="cuda")
    prefix = os.path.join(RESULTS_DIR, f"{mode}_T{seq_len}")

    # warmup
    train_step(model, idx, targets)
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.memory._record_memory_history(max_entries=100_000)
    train_step(model, idx, targets)
    torch.cuda.memory._dump_snapshot(f"{prefix}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"{mode}: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, saved {prefix}.pickle")

    def on_trace_ready(p):
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        p.export_chrome_trace(f"{prefix}.json")
        print(f"saved {prefix}.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=steps),
        on_trace_ready=on_trace_ready,
    ) as p:
        for _ in range(2 + steps):
            train_step(model, idx, targets)
            p.step()


def check():
    """check matching across baseline / checkpointing / offloading (dropout included)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(vocab_size=512, d_model=128, n_head=4, n_layer=4, max_seq_len=256, dropout=0.1)
    torch.manual_seed(0)
    model = Transformer(cfg).to(device).train()
    idx, targets = torch.randint(cfg.vocab_size, (2, 2, 256), device=device)

    results = {}
    for mode in ("baseline", "ckpt", "offload"):
        model.use_checkpoint = mode == "ckpt"
        model.use_offload = mode == "offload"
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1)  # same dropout masks in all runs
        _, loss = model(idx, targets)
        loss.backward()
        results[mode] = loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()}

    loss_b, grads_b = results["baseline"]
    ok = True
    for mode in ("ckpt", "offload"):
        loss_m, grads_m = results[mode]
        max_diff = max((grads_b[n] - grads_m[n]).abs().max().item() for n in grads_b)
        ok &= abs(loss_b - loss_m) < 1e-5 and all(
            torch.allclose(grads_b[n], grads_m[n], atol=1e-5, rtol=1e-4) for n in grads_b
        )
        print(f"device={device} loss baseline={loss_b:.6f} {mode}={loss_m:.6f} max grad diff={max_diff:.2e}")
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only run the gradient correctness check")
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=3, help="active steps of the time profile")
    parser.add_argument("--gpu", type=str, default="0", help="physical GPU index to run on")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.check:
        check()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for mode in ("baseline", "ckpt", "offload"):
        profile_mode(mode, args.seq_len, args.steps)


if __name__ == "__main__":
    main()
