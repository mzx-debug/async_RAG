# async_rag_pipeline.py / run_comparison.py 运行说明

## 1. 当前整体结构

`run_comparison.py` 只是驱动器。它顺序运行：

```text
serial -> async_plain -> async_bucket
```

真正的调度和执行逻辑都在 `async_rag_pipeline.py`。

系统仍然包含三个阶段：

1. `Embedding`
2. `Retrieval`
3. `Generation`

但 `async_bucket` 已经不再是“预打包全部 batch 再派发”，而是在线生成 dispatch plan。

## 2. 当前 async_bucket 的真实工作流

### 2.1 启动前

启动后会先做这些准备：

1. 加载 queries
2. 用 embedding tokenizer 预计算每条 query 的 token 长度
3. 按 token 长度分成 `short / mid / long` 三桶
4. 把每条 query 放入对应 bucket 的待处理池

这一阶段不再预先构造所有 microbatch。

### 2.2 运行时

主线程每次 dispatch 前都会在线决定：

1. 发哪个 bucket
2. 这次 batch size 取多大
3. 这批使用什么 `xE / xR`
4. 这批具体包含哪些 query

然后把这个 dispatch plan 投入流水线。

### 2.3 反馈

每个 batch 完成后，scheduler 会回写运行时 EMA：

- `ema_emb_ms_per_query[(bucket, xE)]`
- `ema_ret_ms_per_query[(bucket, xR)]`
- `ema_gen_ms_per_query[bucket]`
- `ema_transfer_ms_per_query[(bucket, xE, xR)]`
- `ema_batch_size_residual_ms_per_query[(bucket, batch_size)]`

这些值会影响后续 dispatch。

## 3. 三种模式的区别

### 3.1 serial

- 固定 batch
- 固定 `xE/xR`
- 严格串行执行 `E -> R -> G`

### 3.2 async_plain

- 固定 batch
- 固定 `xE/xR`
- 三线程流水线

```text
main -> q_er -> embed_worker -> q_rg -> retrieval_worker -> q_out -> generation_worker
```

### 3.3 async_bucket

- token 长度驱动分桶
- 在线决定 bucket
- 在线决定 batch size
- 在线决定 `xE/xR`
- 在线更新分阶段 EMA

因此它已经是一个在线调度器，不再只是“动态挑桶”。

## 4. 当前有效的调度参数

当前仍然有效的调度参数：

- `--length-short-threshold`
- `--length-long-threshold`
- `--bucket-batch-short`
- `--bucket-batch-mid`
- `--bucket-batch-long`
- `--embed-long-gpu-threshold`
- `--retrieve-gpu-batch-threshold`
- `--backpressure-high`
- `--scheduler-ema-alpha`

说明：

- `bucket-batch-*` 现在是“初始 batch size 基线”，不是最终固定值
- `scheduler-ema-alpha` 控制在线反馈更新速度

## 5. 已删除或废弃的旧语义

以下旧语义已经移除：

- query 压缩比例 `r`
- `_compress_query()`
- `_preprocess_query()`
- `length-hard-threshold`
- `max-processed-length`
- `embed-mid-gpu-threshold`
- `backpressure-low`
- 预先为整个 bucket 固定 action
- 仅用总延迟 EMA 做 action 评分

## 6. 设备迁移与当前接口

当前实现已经开始优化 `xE=1, xR=1` 的路径：

- 如果 embedding 和 retrieval 都在 GPU，embedding 会尽量保留 GPU resident 输出
- retrieval 会优先直接消费 GPU tensor

所以：

- `xE=1, xR=1` 不再默认强制走 `GPU -> CPU -> GPU`
- `xE=1, xR=0` 仍然需要显式回 CPU
- `xE=0, xR=1` 仍然需要把输入送到 GPU retrieval

当前迁移成本既有：

- 隐式阶段观测
- 显式 `transfer` EMA 近似建模

因此它已经比“只靠总延迟学迁移成本”更细，但还不是完整的传输管线建模。

## 7. 如何理解当前打分

现在有两层代价：

### 7.1 action cost

用于选择 `xE/xR`：

```text
embedding + retrieval + transfer
```

### 7.2 dispatch cost

用于比较不同 bucket / batch size：

```text
embedding + retrieval + transfer + generation + batch_size_residual
```

所以：

- generation 不会掩盖 `xE/xR` 的 credit assignment
- batch size 自己也有单独残差项

## 8. 一条典型命令

```powershell
python .\async_rag_pipeline.py `
  --pipeline-mode async_bucket `
  --index-path .\indexes\ivf4096_flat\faiss.index `
  --corpus-path .\data\corpus.jsonl `
  --generator-model meta-llama/Llama-3.1-8B-Instruct `
  --queries-file .\data\queries.jsonl `
  --sample-queries 256 `
  --b 64 --xE 1 --xR 0 `
  --nprobe 128 --topk 1 `
  --scheduler-ema-alpha 0.25 `
  --output-json .\output\summary_async_bucket.json
```

## 8.1 消融实验

`run_ablation.py` 会自动运行一组命名 variant，例如：

- `plain_b16`
- `plain_b64`
- `bucket_fixed_batch_fixed_action`
- `bucket_online_batch_fixed_action`
- `bucket_online_batch_online_action`
- `bucket_online_batch_online_action_no_chunk`

底层通过这些显式开关控制：

- `--ablate-bucketing`
- `--ablate-online-batch`
- `--ablate-online-action`
- `--ablate-chunking`

输出：

- `ablation_rows.json`
- `ablation_table.md`
- 各 variant 的 `summary_<name>.json`

## 9. 当前边界

这版已经实现了：

- tokenizer 长度缓存
- 在线 bucket / batch size / action 联合决策
- GPU-resident `xE=1, xR=1` 路径优化
- 分阶段 EMA 反馈

仍然没有做的是：

- 更精确的设备迁移显式测量
- 更高级的全局最优规划
- 基于真实 tokenized chunk 数的更细粒度长 query 代价模型
