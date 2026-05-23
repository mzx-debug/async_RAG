# Session Handoff (2026-05-16)

**最新更新：2026-05-23 — 研究方向切换到资源受限场景**

---

## 0. 2026-05-23 重大变更：研究方向切换

### 发生了什么

经代码分析确认，在 GPU 显存充裕的环境下，`async_bucket` 的调度器几乎没有优化空间（最优策略固定为 `xE=1, xR=0, b=64`），导致 `async_bucket` 无法显著超过 `plain_b64`。

### 决策

**暂停** `generation_target_v1` 的验证工作（未完成但不是当前优先目标）。

**切换**研究方向到**资源受限场景**：主动限制 GPU 显存（降低 `gpu_memory_utilization`），使 `xE/xR/batch_size` 的联合决策成为有意义的优化问题。

### 已完成的代码改动

见 `docs/session_code_changes.md` 第 0 节 "2026-05-23 资源受限场景调度重构"。

### 接下来要做什么

见 `docs/session_todo.md` 的"立即"和"短期"部分。核心是：

1. 先跑一次 `--disable-memory-aware-scheduling` 的 baseline
2. 再跑一次默认（显存感知启用）对比

---



## 1. Current Goal

Current research direction is now:

- optimize **throughput only**
- treat **large generation batch size** as the primary objective
- deprioritize answer quality concerns for now

This means:

- prompt quality is not the current focus
- chunking is not being optimized as a throughput feature
- bucket logic is no longer assumed to be the center of the scheduler

## 2. Strong Conclusions Already Established

These conclusions are supported by the current comparison / ablation results:

1. The strongest throughput gains come from **larger generation batches**.
2. `plain_b64` is currently the strongest practical baseline.
3. Existing `legacy_bucket` style scheduling has **not** beaten `plain_b64`.
4. Online batch sizing and online action selection have not shown clear gains yet.
5. Scheduler overhead itself is very small and is **not** the bottleneck.

## 3. Files That Matter Most

Core code:

- `E:\R1\async_rag_pipeline_v0\async_rag_pipeline.py`
- `E:\R1\async_rag_pipeline_v0\run_comparison.py`
- `E:\R1\async_rag_pipeline_v0\run_ablation.py`
- `E:\R1\async_rag_pipeline_v0\run_generation_target_eval.py`

Current high-level summaries:

- `E:\R1\async_rag_pipeline_v0\docs\progress_summary_2026-05-16.md`
- `E:\R1\async_rag_pipeline_v0\docs\pipeline_execution_guide.md`
- `E:\R1\async_rag_pipeline_v0\docs\session_code_changes.md`

## 4. Code State Right Now

### 4.1 Stable / validated path

The following path is already exercised by experiments:

- `legacy_bucket` scheduler path
- full tracing:
  - `dispatch_trace`
  - `feedback_trace`
  - `chunk_trace`
  - `timing_breakdown`

### 4.2 New but not yet validated path

A new scheduler mode has already been added to code:

- `--scheduler-mode-choice generation_target_v1`

Intent of this mode:

1. estimate generation target batch range first
2. make device plan second
3. apply optional batch shaping third

Important:

- this path is **implemented in code**
- but has **not yet been experimentally validated**
- do not treat it as a confirmed improvement yet

### 4.3 New trace groups already added for the new path

The code already supports these summary fields:

- `generation_target_trace`
- `device_plan_trace`
- `batch_shaping_trace`

### 4.4 Optional shaping switch

The new path also supports:

- `--enable-batch-shaping`

This should be treated as a secondary optional optimization, not the main scheduler objective.

## 5. Reliability Caveat

There was a real earlier bug in `run_ablation.py`:

- `plain_b16` and `plain_b64` were accidentally routed through `async_bucket`

This has already been corrected in the current local code.

So:

- older ablation conclusions before that fix should not be trusted
- newer ablation conclusions after the fix are the ones to use

## 6. What Still Needs Validation

The next concrete validation target is:

- compare `generation_target_v1` directly against `plain_b64`

Specifically test:

1. `plain_b64_baseline`
2. `generation_target_v1_no_shaping`
3. `generation_target_v1_with_shaping`

## 7. Exact Next Experiment Command

After syncing files to the Linux server, run:

```bash
python ./run_generation_target_eval.py \
  --workdir ~/async_rag_pipeline_v0 \
  --index-path ~/async_rag_pipeline_v0/indexes/ivf_large/faiss.index \
  --corpus-path ~/async_rag_pipeline_v0/data/msmarco-passage-corpus \
  --generator-model /data/home/mazhenxiang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --queries-file ./data/queries_generated.jsonl \
  --sample-queries 512 \
  --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-id 5 \
  --output-dir ./generation_target_eval_large
```

Expected outputs:

- `generation_target_eval_large/generation_target_eval_table.md`
- `generation_target_eval_large/generation_target_eval_rows.json`
- `generation_target_eval_large/summary_plain_b64_baseline.json`
- `generation_target_eval_large/summary_generation_target_v1_no_shaping.json`
- `generation_target_eval_large/summary_generation_target_v1_with_shaping.json`

## 8. Files To Sync To Server

Minimum required set:

1. `E:\R1\async_rag_pipeline_v0\async_rag_pipeline.py`
2. `E:\R1\async_rag_pipeline_v0\run_generation_target_eval.py`
3. `E:\R1\async_rag_pipeline_v0\run_ablation.py`
4. `E:\R1\async_rag_pipeline_v0\README.md`
5. `E:\R1\async_rag_pipeline_v0\docs\progress_summary_2026-05-16.md`
6. `E:\R1\async_rag_pipeline_v0\docs\session_handoff_2026-05-16.md`

## 8.1 Presentation Artifact

A Chinese PPT focused only on `async_bucket` has been generated:

- `E:\R1\async_rag_pipeline_v0\async_bucket_progress_report.pptx`

Current positioning of this deck:

- audience: advisor / group meeting
- focus: mechanism, experiment results, current conclusions
- excludes: detailed original baseline pipeline introduction

## 9. What To Check When Results Come Back

For the three generation-target evaluation outputs, inspect:

1. `wall_throughput_qps`
2. `avg_generation_ms`
3. `generation_target_trace`
4. `device_plan_trace`
5. `batch_shaping_trace`
6. `action_counts`
7. `timing_breakdown`

Key question:

- can `generation_target_v1` beat or at least match `plain_b64`?

If not, the new scheduler direction should probably be reconsidered before more complexity is added.

## 10. Recommended Restart Context

If resuming later, the fastest way to restart is:

1. open this handoff file
2. open `progress_summary_2026-05-16.md`
3. review `async_bucket_progress_report.pptx` if the next discussion is presentation-focused
4. sync the listed files to the Linux server if the next discussion is experiment-focused
5. run `run_generation_target_eval.py`
6. analyze the three resulting summaries against `plain_b64`
