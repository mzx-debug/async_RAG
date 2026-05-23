# Results Update: Resource-Constrained Scheduling Refactor

Date: 2026-05-23

## What Changed

The codebase has been refactored to target **resource-constrained scenarios** instead of the previously-tested GPU-memory-abundant server environment.

The core research hypothesis:

> When GPU memory is abundant, `xE=1, xR=0, b=64` is the fixed optimum and the scheduler has no meaningful decision space. Switching to resource-constrained scenarios should restore optimization room for the scheduler.

## Code Changes

### New: `ResourceTracker` class

Monitors GPU memory in real time via `torch.cuda.mem_get_info()` and estimates per-stage memory costs.

Key methods:
- `pressure_level(gpu_mem_gb)` → `"high" | "medium" | "low"`
- `max_batch_size_for_action(x_e, x_r)` → max feasible batch size under current memory
- `estimate_embed_activations_gb()`, `estimate_generation_cost_gb()` → per-batch memory estimates

### New: Memory-aware action selection (`_action_feasible`)

Replaces the old hardcoded `gpu_mem_gb < 20.0` threshold with a dynamic, fine-grained approach:

```
Old:  if x_r == 1 and gpu_mem_gb < 20.0: continue  # hard filter
New:  if batch_size > max_batch_size_for_action(x_e, x_r): return False
      penalty = pressure_penalty(gpu_mem_gb, x_e, x_r)   # soft score adjustment
```

### New: Memory-pressure-aware bucket priority (`_bucket_priority`)

Under high memory pressure, long queries are dispatched first (they consume the most memory, so dispatching them early frees memory earlier).

Under low memory pressure, short queries stay highest priority (追求高 throughput).

### New: Lookahead dispatch

The old behavior: dispatch one batch → wait for generation feedback → dispatch next.

The new behavior: push N batches ahead of the generation stage without waiting. The value of N is driven by memory pressure:

| Pressure | Trigger | Max Lookahead |
|---------|---------|--------------|
| high   | q_rg >= 1 | 3 batches |
| medium | q_rg >= 2 | 1-2 batches |
| low    | q_rg >= 4 | 1 batch |

This creates real pipeline overlap: CPU embed (xE=0) runs in parallel with GPU retrieve + GPU generate.

### New CLI flags

```
--enable-memory-aware-scheduling   (default: True; use --disable to turn off)
--gpu-mem-low-threshold-gb        (default: 4.0 GiB)
--gpu-mem-medium-threshold-gb     (default: 10.0 GiB)
--gpu-mem-high-batch-penalty      (default: 50.0 ms)
--enable-lookahead-dispatch       (default: off; must be passed explicitly)
--faiss-index-gb                 (default: 2.0 GiB)
```

## Backward Compatibility

All new parameters have `getattr(..., default)` guards. Existing runner scripts (`run_comparison.py`, `run_ablation.py`) work without modification.

`--enable-memory-aware-scheduling` defaults to True. Pass `--disable-memory-aware-scheduling` to fall back to old static-threshold behavior for comparison.

## Expected Outcome

**Under resource-constrained conditions** (e.g., `--gpu-memory-utilization 0.3`), the memory-aware scheduler should:

1. Choose `xE=0, xR=0` more often under high memory pressure
2. Prioritize long queries when GPU memory is tight
3. Use lookahead dispatch to keep the pipeline saturated

**Key experiment to run first**:

```powershell
# Baseline (old behavior)
python .\async_rag_pipeline.py `
  --pipeline-mode async_bucket `
  --gpu-memory-utilization 0.3 `
  --b 16 --xE 1 --xR 0 `
  --disable-memory-aware-scheduling `
  --index-path ... --corpus-path ... --generator-model ... `
  --queries-file .\data\queries_generated.jsonl `
  --output-json .\output\baseline_0.3.json

# Memory-aware (new behavior)
python .\async_rag_pipeline.py `
  --pipeline-mode async_bucket `
  --gpu-memory-utilization 0.3 `
  --b 16 --xE 1 --xR 0 `
  --index-path ... --corpus-path ... --generator-model ... `
  --queries-file .\data\queries_generated.jsonl `
  --output-json .\output\mem_aware_0.3.json

# Memory-aware + lookahead
python .\async_rag_pipeline.py `
  --pipeline-mode async_bucket `
  --gpu-memory-utilization 0.3 `
  --b 16 --xE 1 --xR 0 `
  --enable-lookahead-dispatch `
  --index-path ... --corpus-path ... --generator-model ... `
  --queries-file .\data\queries_generated.jsonl `
  --output-json .\output\mem_aware_lookahead_0.3.json
```

Compare `wall_throughput_qps`, `action_counts`, and `bucket_counts` across the three runs.

## Key Files to Reference

- `docs/session_todo.md` — immediate and short-term tasks
- `docs/session_experiments.md` — full experiment designs
- `docs/session_code_changes.md` — detailed code change log
- `docs/session_research_progress.md` — research motivation and direction
