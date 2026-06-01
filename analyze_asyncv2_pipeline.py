#!/usr/bin/env python3
"""Analyze gen_per_token from async_v2 pipeline output."""
import json
from pathlib import Path

OUT = Path("/home/cloudteam/rag_mzx/output/pipeline_comparison")
data = json.load(open(OUT / "asyncv2.json"))
pb = data.get("per_batch", [])

print("async_v2 pipeline breakdown (B=32 initial, adaptive B):")
print(f"  Total wall_time_ms: {data['wall_time_ms']:.0f}")
print(f"  Total queries: {sum(b.get('batch_size',0) for b in pb)}")
print()
print(f"  {'Batch':>6} | {'B':>5} | {'gen_sec':>10} | {'gen_tok':>8} | {'gen_ms/q':>10}")
print(f"  {'-'*50}")
total_wall = data["wall_time_ms"]
for b in pb:
    idx = b.get("batch_index", "?")
    bs = b.get("batch_size", 0)
    gs = b.get("generation_sec", 0)
    gt = b.get("generated_tokens", 0)
    gms_q = gs * 1000 / bs if bs else 0
    print(f"  {idx:>6} | {bs:>5} | {gs:>10.3f} | {gt:>8} | {gms_q:>10.2f}")

print()
print("Key insight:")
print(f"  3-stage pipeline overlaps Emb+Ret+Gen across batches")
print(f"  B=256 batch: generation overlaps with B=32/8/4 batches arriving")
print(f"  wall_time = sum(gen_sec per batch) = {sum(b.get('generation_sec',0) for b in pb):.3f}s")
print(f"  Actual wall_time = {total_wall:.0f}ms = {total_wall/1000:.3f}s")
print(f"  Difference: {total_wall/1000 - sum(b.get('generation_sec',0) for b in pb):.3f}s (queue/setup overhead)")
