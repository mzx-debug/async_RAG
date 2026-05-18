# 代码改动记录

所有改动均在 `async_rag_pipeline.py`，除非特别说明。

---

## 0. 2026-05-16 参数收口与死代码清理

**目标**：删除“名义可配、实际不生效”的僵尸参数与伪动态逻辑，让 CLI、summary 和真实行为一致。

**删除的 CLI 参数**：

- `--length-hard-threshold`
- `--max-processed-length`
- `--embed-mid-gpu-threshold`
- `--backpressure-low`

**删除的死代码 / 伪语义**：

- `BatchStats.r`
- `ScheduledMicrobatch.processed_queries`
- `ScheduledMicrobatch.estimated_load`
- query 压缩相关的 `_compress_query()` / `_preprocess_query()`
- 未生效的 per-run 动态 batch size 状态与反馈更新
- summary 中的 `scheduler.bucket_thresholds.hard`
- action key 中的 `r=...`

**保留并明确化的行为**：

- `async_bucket` 仍然按三桶预分批
- 每个桶仍然在批次准备阶段静态选择 `xE/xR`
- 运行时只动态决定“下一次先派发哪个桶”

**同步文档**：

- 重写 `README.md`
- 重写 `docs/pipeline_execution_guide.md`

---

## 0.1 2026-05-16 在线 EMA 调度反馈

**目标**：在保留三桶预分批结构的前提下，把 `async_bucket` 改成“在线重选 action + 运行时反馈修正”的调度器。

**新增能力**：

- 每个 microbatch dispatch 前重新评估 `xE/xR`
- 每个 bucket dispatch 前综合 `q_er / q_rg` 压力和运行时 EMA 代价
- 每个 batch 完成后分别回写 embedding / retrieval / generation 三类 EMA
- generation EMA 显式进入 bucket dispatch 打分

**新增参数**：

- `--scheduler-ema-alpha`

**保留不变的结构**：

- 三桶静态分桶
- LPT 打包
- 固定 `bucket_batch_short/mid/long`

**不再使用的旧做法**：

- 运行前一次性固定所有 `async_bucket` 的 action
- 用总延迟 EMA 直接驱动 action 选择
- per-run 动态 batch size 反馈

---

## 0.2 2026-05-16 最终在线工作流重构

**目标**：把 scheduler 从“预打包 + 动态挑桶”升级成真正的在线 dispatch 系统。

**核心变化**：

- query 长度改为 tokenizer token 长度，并在加载 queries 后一次性缓存
- scheduler 不再预构造全部 microbatch，而是维护每个 bucket 的待处理 query 池
- 每次 dispatch 时同时决定：
  - bucket
  - batch size
  - `xE/xR`
  - 本批 query 组成

**接口变化**：

- `ScheduledMicrobatch` 新增 `query_indices` 和 `token_lengths`
- `EmbeddingPayload` 改为同时支持 `embeddings_cpu` / `embeddings_gpu`
- `RetrievalStage` 改为同时支持 CPU numpy 和 GPU tensor 输入

**性能相关变化**：

- `xE=1, xR=1` 时尽量保留 GPU-resident embedding，不再默认 `GPU -> CPU -> GPU`
- 增加 `transfer` EMA 和 `batch_size residual` EMA
- batch size 从固定常量改为候选集合中的在线选择

**当前状态**：

- 已具备在线 bucket / batch size / action 联合决策
- 仍然使用启发式 + EMA，而不是全局最优规划

---

## 1. stage_cache 预填充（Bug Fix）

**问题**：`embed_worker` 和 `retrieval_worker` 的 `stage_cache` 初始化为空字典，导致每次 async pipeline 运行时第一个 microbatch 都会重新调用 `AutoModel.from_pretrained` 和 `faiss.read_index`，造成 async_bucket 的 embedding 耗时从 4.78ms 暴涨到 39.84ms。

**修改位置**：
- `embed_worker` 内
- `retrieval_worker` 内

**修改内容**：
```python
# 修前
stage_cache: Dict[str, QueryEmbeddingStage] = {}
stage_cache: Dict[str, RetrievalStage] = {}

# 修后
stage_cache: Dict[str, QueryEmbeddingStage] = {self.embedding_backend: self.embedding_stage}
stage_cache: Dict[str, RetrievalStage] = {self.retrieval_backend: self.retrieval_stage}
```

**效果**：worker 直接复用 `__init__` 里已加载的 stage 对象，只有切换 backend 时才新建。

---

## 2. Warmup 机制（公平性修复）

**问题**：三种模式用 subprocess 依次运行，第一个模式冷启动，后两个热启动，对比不公平。

**修改内容**：
```python
def _warmup(self) -> None:
    """Run one small batch through E→R→G to warm up CUDA kernels and model caches."""
    warmup_queries = self.queries[:min(4, len(self.queries))]
    self.logger.info("Warming up pipeline with %d queries (results discarded)...", len(warmup_queries))
    embeddings, _ = self.embedding_stage(warmup_queries)
    retrieved_docs, _ = self.retrieval_stage(embeddings)
    self.generation_stage(warmup_queries, retrieved_docs)
    if self.gpu_available:
        torch.cuda.synchronize()
    self.logger.info("Warmup complete.")
```

**效果**：三种模式计时起点状态一致，CUDA kernel、模型权重缓存、FAISS index 均已热启动。

---

## 3. 长 Query 切分 Embedding（功能改进）

**问题**：原来对超长 query 的处理是截断，会丢失 query 后半段的语义信息。

**修改内容**：
```python
# 按 64 token 切分（overlap=16 token）
chunk_size = 64
overlap = chunk_size // 4
step = chunk_size - overlap

# 展平所有 chunk，一次 forward pass
# 按 query 分组，多 chunk 取平均，再 L2 归一化
```

**作用域限制**：切片嵌入仅在 `async_bucket` 模式下启用（`chunked_embedding=True`）。serial 和 async_plain 使用简单 truncation，保证对比公平性。

**效果**：
- ≤64 token 的 query：零额外开销
- >64 token 的 query（仅 async_bucket）：切成多个 chunk 分别 embed，取平均向量，不丢失信息
- serial/async_plain：超过 max_length 的 token 直接截断

---

## 4. Arrow 格式语料库支持（功能扩展）

**修改内容**：
```python
if path.is_dir() and (path / "state.json").exists():
    from datasets import load_from_disk
    return load_from_disk(str(path))
```

---

## 5. 调度器 action 空间改进（迭代优化）

**历史**：最初 `available_actions` 包含 8 种 action（含 r=0.8 变体），导致调度器选到不可行的 xR=1 卡死。第一次修复是用 CLI 参数过滤（`a["xE"] <= cli_xE and a["xR"] <= cli_xR`）。

**当前方案**：开放完整 action 空间，改用运行时条件过滤：
```python
# 调度器拥有完整 action 空间（4 种，r 固定为 1.0）
self.available_actions = [
    {"xE": 0, "xR": 0, "r": 1.0},
    {"xE": 1, "xR": 0, "r": 1.0},
    {"xE": 0, "xR": 1, "r": 1.0},
    {"xE": 1, "xR": 1, "r": 1.0},
]

# _choose_action_for_batch 中运行时过滤：
# - gpu_available=False → 排除 xE=1 和 xR=1
# - gpu_free_mem < 20.0 GiB → 排除 xR=1
# - batch_size < retrieve_gpu_batch_threshold → 排除 xR=1
```

**关于 r=0.8**：已从 action 空间中移除。所有 query 以原始长度处理，不做截断压缩。`_choose_action_for_batch` 中 ratio 相关的评分逻辑仍保留但不会触发（action 空间中无 r<1.0 的选项）。

**配套改动**：
- 新增 `_detect_gpu_free_memory_gb()` 方法，使用 `torch.cuda.mem_get_info()` 获取实时可用显存
- `prepare_bucket_batches` 传入可用显存而非总显存
- 显存阈值从 8.0 GiB 提高到 20.0 GiB（FAISS GPU 索引需要大量显存）

---

## 6. 删除 short 桶 GPU embedding 禁用规则（严重 Bug Fix）

**问题**：short 桶（l_max≤48）被强制 CPU embedding，在 async 模式下耗时 1400-2100ms/批（GPU 只需 60ms），且受 GIL 争抢影响。

**修改内容**：
```python
# 删除此过滤条件
if x_e == 1 and l_max < self.embed_mid_gpu_threshold and l_avg < self.embed_mid_gpu_threshold:
    continue
```

**效果**：所有桶都可以使用 GPU embedding，short 桶 embedding 从 ~2000ms 降到 ~60ms。

---

## 7. FAISS 多线程 + 大 batch 分片检索

**问题**：FAISS 默认单线程，大库 nprobe=128 检索极慢；大 batch 无中间日志，看起来像卡死。

**修改内容**：
```python
# RetrievalStage.__init__ 中
n_threads = os.cpu_count() or 8
self.faiss.omp_set_num_threads(n_threads)

# RetrievalStage.__call__ 中
chunk_size = 16
if n_queries > chunk_size:
    # 分片检索，每片打日志
    for i in range(0, n_queries, chunk_size):
        chunk = embeddings[i : i + chunk_size]
        _, idx_chunk = self.index.search(chunk, self.topk)
        self.logger.info("  retrieval chunk %d-%d/%d done (%.1fms)", ...)
```

---

## 8. tqdm 进度条 + 改进日志

**修改内容**：
- 添加 `from tqdm import tqdm`
- serial 模式：tqdm 进度条 + 每批显示 ret/gen 耗时 + ETA
- async 模式：tqdm 进度条（generation_worker 更新）+ 三个 worker 各自打日志
- `--log-interval` 默认从 10 改为 1

**修改文件**：`run_comparison.py`
- 去掉 `capture_output=True`，子进程输出直接透传到终端

---

## 9. 边缘设备场景的 FAISS 线程策略（性能/稳定性改进）

**背景**：原实现固定 `faiss.omp_set_num_threads(os.cpu_count())`，会占满全部逻辑核。在边缘设备场景下，这容易挤占系统和 Python worker（embed/retrieval/generation 线程）资源，导致整体抖动、发热和响应变差。

**修改内容**：
1. `RetrievalStage` 新增参数 `faiss_omp_threads`。
2. 新增命令行参数：
   - `--faiss-omp-threads`（默认 `0`）
   - `0` 表示自动“边缘友好”策略；`>0` 表示强制指定线程数。
3. 自动策略（按逻辑核）：
   - `>=16` 核：`4` 线程
   - `8-15` 核：`3` 线程
   - `4-7` 核：`2` 线程
   - `<4` 核：`1` 线程
4. 将该参数贯通到：
   - `StandaloneRAGPipeline.__init__`
   - `_switch_backends_if_needed`
   - `_build_retrieval_stage`

**效果**：
- 默认不再吃满 CPU，更符合边缘设备“吞吐/温控/系统可用性”平衡目标。
- 仍保留手动调优入口，可按设备规格覆盖线程数。

---

## 10. 动态 Batch Size（延迟反馈调整）

**问题**：batch_size 固定（short=64, mid=32, long=16），仅有简单的背压缩放，无法适应不同硬件和负载下的最优值。

**修改内容**：

1. `GreedyBucketScheduler.__init__` 新增 `_batch_state` 字典，每个桶独立跟踪 size/direction/latency：
```python
self._batch_state = {
    "short": {"size": 64, "last_latency_per_query": None, "direction": 1},
    "mid":   {"size": 32, "last_latency_per_query": None, "direction": 1},
    "long":  {"size": 16, "last_latency_per_query": None, "direction": 1},
}
self._batch_min = 8
self._batch_max = 128
self._batch_step = 8
```

2. 新增 `get_dynamic_batch_size(bucket)` 和 `update_batch_feedback(bucket, total_sec, batch_size)` 方法

3. `_bucket_batch_size` 改为读取动态值，背压作为安全阀

4. serial 模式和 async_bucket generation_worker 中每个 batch 完成后调用 `update_batch_feedback`

**算法**：hill-climbing，延迟下降则保持方向，延迟上升则反转方向，步长固定 8。

---

## 11. Per-Batch Action 记录

**问题**：`per_batch` 输出中只有延迟数据，无法看到调度器对每个 batch 选了什么 action。

**修改内容**：

`BatchStats` dataclass 新增字段：
```python
xE: int = 0
xR: int = 0
r: float = 1.0
```

两处 `BatchStats(...)` 构造时传入 `microbatch.action` 中的值。

**效果**：输出 JSON 中每个 batch 记录包含完整决策信息：
```json
{
  "batch_index": 1,
  "batch_size": 61,
  "bucket": "short",
  "embedding_sec": 0.053,
  "retrieval_sec": 1.237,
  "generation_sec": 4.074,
  "generated_tokens": 7808,
  "xE": 1,
  "xR": 0,
  "r": 1.0
}
```

| 文件 | 说明 |
|------|------|
| `generate_queries.py` | 从 msmarco_2k.jsonl 生成 512 条问题集（short/mid/long = 50%/35%/15%） |
| `generate_long_queries.py` | 生成 100 条超长问题集（128-226 token），用于测试切分逻辑 |
| `data/queries_generated.jsonl` | 512 条问题集 |
| `data/queries_long.jsonl` | 100 条超长问题集 |
| `comparison/analysis_report.md` | 小库实验详细分析报告 |
| `comparison_large/summary_*.json` | 大库实验结果 |
| `docs/session_research_progress.md` | 研究进展记录 |
| `docs/session_code_changes.md` | 本文件 |
| `docs/session_experiments.md` | 实验记录 |
| `docs/session_todo.md` | 待办事项 |
