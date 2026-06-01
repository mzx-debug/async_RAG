#!/usr/bin/env python3
"""Analyze existing gen_token_sweep results."""
import json, statistics
from pathlib import Path

OUT = Path("/home/cloudteam/rag_mzx/output/gen_token_sweep")

def extract(data):
    pb = data.get("per_batch", [])
    total_gen_ms = sum(b.get("generation_sec", 0) * 1000 for b in pb)
    total_tokens = sum(b.get("generated_tokens", 0) for b in pb)
    n_batches = len(pb)
    if total_tokens == 0:
        return {}
    gen_pt = total_gen_ms / total_tokens
    avg_out = total_tokens / n_batches if n_batches else 0
    return {"gen_per_token": gen_pt, "avg_output": avg_out, "total_gen_ms": total_gen_ms,
            "total_tokens": total_tokens, "n_batches": n_batches}

results = {}
for gpu in ["0.3", "0.5", "0.8"]:
    for b in [8]:
        path = OUT / f"gpu{gpu}_b{b}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        params = extract(data)
        results[float(gpu)] = params
        print(f"gpu={gpu}: gen_per_token={params.get('gen_per_token',0):.4f} ms/tok, "
              f"avg_out={params.get('avg_output',0):.1f}, batches={params.get('n_batches',0)}")

print("\ngen_per_token by gpu_util:")
print("  gpu_util | gen_pt  | avg_out")
for gpu, p in sorted(results.items()):
    print(f"  {gpu:>8.1f} | {p['gen_per_token']:>6.4f} | {p['avg_output']:>7.1f}")
