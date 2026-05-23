# 资源受限场景测试指南

## 1. 核心参数说明

| 参数 | 说明 | 资源受限测试建议值 |
|------|------|------------------|
| `--gpu-memory-utilization` | vLLM 显存占用比例 | `0.2` ~ `0.4` |
| `--b` | CLI 指定的 batch size | `16` (调度器会动态调整) |
| `--xE` | CLI 指定的 embedding 设备 | `1` (调度器会动态选择) |
| `--xR` | CLI 指定的 retrieval 设备 | `0` (调度器会动态选择) |
| `--enable-memory-aware-scheduling` | 显存感知调度开关 | 默认开启 |

## 2. Action 空间

```
xE=0, xR=0  → CPU embed + CPU retrieval  (释放最多 GPU 显存)
xE=1, xR=0  → GPU embed + CPU retrieval  (默认选择)
xE=0, xR=1  → CPU embed + GPU retrieval  (GPU retrieval 受限)
xE=1, xR=1  → GPU embed + GPU retrieval  (显存最紧张)
```

## 3. 显存感知调度逻辑

### 3.1 显存压力等级

```
high:   GPU 可用显存 < 4 GiB
medium: GPU 可用显存 < 10 GiB
low:    GPU 可用显存 >= 10 GiB
```

### 3.2 压力感知行为

| 压力等级 | Action 选择倾向 | Bucket 优先级 |
|---------|----------------|---------------|
| high | 倾向 xE=0,xR=0 | long > mid > short |
| medium | 平衡选择 | long 适度优先 |
| low | 默认 xE=1,xR=0 | short > mid > long |

## 4. 实验命令

### 4.1 创建输出目录

```bash
mkdir -p ./output/resource_test
```

### 4.2 对照组：静态策略

#### 对照组 A: 固定 b=64 (显存充裕时的最强基线)

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_plain \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --queries-file ./data/queries_generated.jsonl \
  --sample-queries 256 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 64 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-memory-utilization 0.3 \
  --output-json ./output/resource_test/plain_b64_util0.3.json
```

#### 对照组 B: 固定 b=16 (显存受限下的实际可行 batch)

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_plain \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --queries-file ./data/queries_generated.jsonl \
  --sample-queries 256 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-memory-utilization 0.3 \
  --output-json ./output/resource_test/plain_b16_util0.3.json
```

#### 对照组 C: CPU embedding 策略 (xE=0, 释放 GPU)

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_plain \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --queries-file ./data/queries_generated.jsonl \
  --sample-queries 256 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 48 --xE 0 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-memory-utilization 0.3 \
  --output-json ./output/resource_test/plain_xE0_b48.json
```

### 4.3 实验组：调度器策略

#### 实验组 A: 显存感知调度器 (核心实验)

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --queries-file ./data/queries_generated.jsonl \
  --sample-queries 256 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-memory-utilization 0.3 \
  --enable-memory-aware-scheduling \
  --output-json ./output/resource_test/bucket_mem_aware.json
```

#### 实验组 B: 长查询集 (验证 long bucket 优先级)

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --queries-file ./data/queries_long.jsonl \
  --sample-queries 100 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-memory-utilization 0.3 \
  --enable-memory-aware-scheduling \
  --output-json ./output/resource_test/bucket_long_queries.json
```

### 4.4 调度器消融实验

#### 消融 A: 关闭显存感知调度

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --queries-file ./data/queries_generated.jsonl \
  --sample-queries 256 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-memory-utilization 0.3 \
  --disable-memory-aware-scheduling \
  --output-json ./output/resource_test/bucket_no_mem_aware.json
```

#### 消融 B: 固定 batch size

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --queries-file ./data/queries_generated.jsonl \
  --sample-queries 256 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-memory-utilization 0.3 \
  --enable-memory-aware-scheduling \
  --ablate-online-batch \
  --output-json ./output/resource_test/bucket_fixed_batch.json
```

#### 消融 C: 固定 action

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --queries-file ./data/queries_generated.jsonl \
  --sample-queries 256 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-memory-utilization 0.3 \
  --enable-memory-aware-scheduling \
  --ablate-online-action \
  --output-json ./output/resource_test/bucket_fixed_action.json
```

### 4.5 显存利用率梯度实验

```bash
# 显存利用率从 0.2 到 0.8，找到调度器收益的临界点
for util in 0.2 0.3 0.4 0.5 0.6 0.8; do
    python ./async_rag_pipeline.py \
      --pipeline-mode async_bucket \
      --index-path ./indexes/ivf4096_flat/faiss.index \
      --corpus-path ./data/corpus.jsonl \
      --queries-file ./data/queries_generated.jsonl \
      --sample-queries 256 \
      --generator-model meta-llama/Llama-3.1-8B-Instruct \
      --b 16 --xE 1 --xR 0 \
      --nprobe 128 --topk 1 \
      --gpu-memory-utilization $util \
      --enable-memory-aware-scheduling \
      --output-json ./output/resource_test/gradient_util${util}.json
done
```

## 5. 结果分析

### 5.1 关键指标

从输出的 JSON 文件中提取以下指标：

```python
# 核心性能指标
wall_throughput_qps   # 实际吞吐量 (最重要)
wall_time_ms          # 实际耗时
avg_emb_ms            # embedding 延迟
avg_ret_ms            # retrieval 延迟
avg_gen_ms            # generation 延迟

# 调度器行为指标
action_counts          # 各 action 被选中的次数
bucket_counts          # 各 bucket 被调度的次数
max_q_er               # embed 队列最大深度
max_q_rg               # retrieval 队列最大深度
```

### 5.2 预期结论

| 场景 | 判断标准 |
|------|---------|
| **调度器有效** | `bucket_mem_aware` 的 QPS > `plain_b16` |
| **xE=0 策略有价值** | `action_counts` 中 `xE0_xR0` 或 `xE0_xR1` 出现 |
| **显存感知生效** | 高显存压力时 `xE0` 被选中次数增加 |
| **调度器无收益** | `action_counts` 全是 `xE1_xR0`，等同于静态策略 |

### 5.3 完整对比表模板

```markdown
| 实验 | wall_QPS | avg_emb_ms | avg_ret_ms | avg_gen_ms | action_counts | bucket_counts |
|------|----------|-----------|-----------|-----------|--------------|--------------|
| plain_b64 | | | | | | |
| plain_b16 | | | | | | |
| plain_xE0 | | | | | | |
| bucket_mem_aware | | | | | | |
```

## 6. 快速运行脚本

创建 `run_resource_test.sh`：

```bash
#!/bin/bash
# run_resource_test.sh

OUTPUT_DIR="./output/resource_test"
mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
    "--index-path" "./indexes/ivf4096_flat/faiss.index"
    "--corpus-path" "./data/corpus.jsonl"
    "--queries-file" "./data/queries_generated.jsonl"
    "--sample-queries" "256"
    "--generator-model" "meta-llama/Llama-3.1-8B-Instruct"
    "--b" "16" "--xE" "1" "--xR" "0"
    "--nprobe" "128" "--topk" "1"
    "--gpu-memory-utilization" "0.3"
)

# 实验 1: plain_b16
python ./async_rag_pipeline.py "${COMMON_ARGS[@]}" \
    --pipeline-mode async_plain \
    --output-json "$OUTPUT_DIR/plain_b16.json"

# 实验 2: plain_xE0_b48
python ./async_rag_pipeline.py "${COMMON_ARGS[@]}" \
    --pipeline-mode async_plain --b 48 --xE 0 --xR 0 \
    --output-json "$OUTPUT_DIR/plain_xE0_b48.json"

# 实验 3: bucket_mem_aware
python ./async_rag_pipeline.py "${COMMON_ARGS[@]}" \
    --pipeline-mode async_bucket --enable-memory-aware-scheduling \
    --output-json "$OUTPUT_DIR/bucket_mem_aware.json"

echo "All experiments completed. Results in $OUTPUT_DIR"
```

运行：

```bash
chmod +x ./run_resource_test.sh
./run_resource_test.sh
```

## 7. 显存感知参数调优

如果默认的显存阈值不适合你的 GPU，可以调整：

```bash
# 低显存 GPU (如 16GB 总显存)
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --gpu-memory-utilization 0.3 \
  --gpu-mem-low-threshold-gb 2.0 \
  --gpu-mem-medium-threshold-gb 6.0 \
  --gpu-mem-high-batch-penalty 30.0 \
  # ... 其他参数 ...
  --output-json ./output/resource_test/tuned.json

# 高显存 GPU (如 40GB 总显存)
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --gpu-memory-utilization 0.3 \
  --gpu-mem-low-threshold-gb 6.0 \
  --gpu-mem-medium-threshold-gb 12.0 \
  --gpu-mem-high-batch-penalty 50.0 \
  # ... 其他参数 ...
  --output-json ./output/resource_test/tuned.json
```
