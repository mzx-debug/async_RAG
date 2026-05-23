# 资源受限场景测试指南 — V1 (BEIR 语料库)

> 本指南适配 V1 版本（资源受限技术栈 + BEIR 真实语料库）。

## 1. 技术栈概览

| 组件 | V0 (服务器) | V1 (受限设备) |
|------|------------|--------------|
| 生成模型 | Llama-3.1-8B-Instruct | Qwen2.5-3B-Instruct |
| Embedding | e5-large-v2 (1024-dim) | all-MiniLM-L6-v2 (384-dim) |
| 语料库 | 880 万条（随机采样） | BEIR 真实语料库（3.6K–8.7K 条） |
| 索引 | IVF4096 (34 GB) | Flat (~1–2 MB) |
| vLLM 显存占用 | 0.3 (模拟) | 0.6 (真实受限) |
| nprobe | 128 | 1 |

## 2. BEIR 数据集选择

| 数据集 | 文档数 | 查询数 | 领域 | 索引大小 | 推荐显存 |
|--------|--------|--------|------|---------|---------|
| `nfcorpus` | ~3,600 | ~323 | 生物医学（饮食/健康） | ~0.7 MB | <= 6GB |
| `scifact` | ~5,180 | ~1,109 | 科学声明检索 | ~1.0 MB | 6–12GB |
| `arguana` | ~8,700 | ~1,409 | 论点检索 | ~1.7 MB | 12+ GB |

所有数据集都包含真实的相关性标注（qrels），检索难度非随机采样可比。

## 3. 核心参数说明

| 参数 | 说明 | V1 默认值 | 说明 |
|------|------|----------|------|
| `--gpu-memory-utilization` | vLLM 显存占用比例 | `0.6` | 3B 模型在 8GB GPU 上可设 0.6–0.8 |
| `--b` | CLI batch size | `32` | 小模型下可设更大 |
| `--xE` | Embedding 设备 | `1` | 调度器动态选择 |
| `--xR` | Retrieval 设备 | `0` | 调度器动态选择 |
| `--nprobe` | FAISS probe 数 | `1` | Flat 索引不需要 nprobe |
| `--enable-memory-aware-scheduling` | 显存感知调度开关 | 默认开启 | |

## 4. Action 空间

```
xE=0, xR=0  → CPU embed + CPU retrieval  (释放最多 GPU 显存)
xE=1, xR=0  → GPU embed + CPU retrieval  (默认选择)
xE=0, xR=1  → CPU embed + GPU retrieval  (小索引可全部加载 GPU)
xE=1, xR=1  → GPU embed + GPU retrieval  (显存最紧张)
```

**V1 特有**：Flat 索引只有 ~1 MB，`xR=1` 在 V1 中变得完全可行——这与 V0 的大索引场景完全不同。

## 5. 实验命令

### 5.1 构建阶段（一次性）

```bash
# 1. 下载 BEIR 语料库（以 nfcorpus 为例）
python corpus_builder.py --dataset nfcorpus --output ./data/beir_nfcorpus

# 2. 构建索引
python build_index.py \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --output-dir ./indexes/beir_nfcorpus \
  --model-path sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 256 --max-length 384 \
  --pooling-method mean --use-fp16 --faiss-type Flat --device cuda

# 3. 后处理查询（添加 bucket 标注）
python generate_queries.py \
  --queries-file ./data/beir_nfcorpus/queries_beir.jsonl \
  --output ./data/beir_nfcorpus/queries.jsonl \
  --tokenizer-model sentence-transformers/all-MiniLM-L6-v2 \
  --auto-threshold
```

### 5.2 对照组：静态策略

#### 对照组 A: plain_b32（默认基线）

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_plain \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 32 --xE 1 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --output-json ./output/resource_test/plain_b32.json
```

#### 对照组 B: plain_b16（受限 batch）

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_plain \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 16 --xE 1 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --output-json ./output/resource_test/plain_b16.json
```

#### 对照组 C: CPU embedding 策略（xE=0，释放 GPU）

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_plain \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 48 --xE 0 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --output-json ./output/resource_test/plain_xE0_b48.json
```

### 5.3 实验组：调度器策略

#### 实验组 A: 显存感知调度器（核心实验）

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 32 --xE 1 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --enable-memory-aware-scheduling \
  --output-json ./output/resource_test/bucket_mem_aware.json
```

#### 实验组 B: 测试 xR=1 可行性（V1 特有实验）

```bash
# xR=1 在 V1 中变得可行，因为 Flat 索引只有 ~1MB
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 32 --xE 1 --xR 1 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --enable-memory-aware-scheduling \
  --output-json ./output/resource_test/bucket_xR1.json
```

### 5.4 消融实验

```bash
# 消融 A: 关闭显存感知调度
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 32 --xE 1 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --disable-memory-aware-scheduling \
  --output-json ./output/resource_test/bucket_no_mem_aware.json

# 消融 B: 固定 batch size
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 32 --xE 1 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --enable-memory-aware-scheduling \
  --ablate-online-batch \
  --output-json ./output/resource_test/bucket_fixed_batch.json

# 消融 C: 固定 action
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 32 --xE 1 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --enable-memory-aware-scheduling \
  --ablate-online-action \
  --output-json ./output/resource_test/bucket_fixed_action.json
```

## 6. 结果分析

### 6.1 关键指标

```python
wall_throughput_qps   # 实际吞吐量（最重要）
wall_time_ms          # 实际耗时
avg_emb_ms            # embedding 延迟
avg_ret_ms            # retrieval 延迟
avg_gen_ms            # generation 延迟
action_counts          # 各 action 被选中的次数
bucket_counts          # 各 bucket 被调度的次数
```

### 6.2 预期结论（V1 特有）

| 判断 | 标准 |
|------|------|
| **调度器在 V1 中仍然有效** | `async_bucket` 的 QPS > `plain_b32` |
| **xR=1 在 V1 中有价值** | `action_counts` 中出现 `xR=1` 且 wall_time 更短 |
| **xE=0 策略有价值** | `action_counts` 中出现 `xE0` |
| **V1 vs V0 差异** | V1 中 retrieval 占比更高，xR=1 更常用 |
| **调度器无收益** | 所有 action 都是 `xE1_xR0`，等同于静态策略 |

## 7. V1 特有研究问题

V1 与 V0 最大的不同在于 **xR=1 的可行性**：

- V0: 34GB 索引，xR=1 几乎不可行
- V1: ~1MB 索引，xR=1 完全可行，且可能比 xR=0 更快

如果 async_bucket 频繁选择 xR=1 并获得更好的 QPS，这说明在小索引场景下，GPU retrieval 确实有优势——这是 V0 无法验证的。

## 8. 显存感知参数调优

```bash
# 低显存 GPU (如 6GB 总显存)
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --gpu-memory-utilization 0.5 \
  --gpu-mem-low-threshold-gb 1.5 \
  --gpu-mem-medium-threshold-gb 3.0 \
  --gpu-mem-high-batch-penalty 30.0 \
  --output-json ./output/resource_test/tuned_low_mem.json

# 高显存 GPU (如 12GB 总显存)
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --gpu-memory-utilization 0.7 \
  --gpu-mem-low-threshold-gb 2.5 \
  --gpu-mem-medium-threshold-gb 6.0 \
  --output-json ./output/resource_test/tuned_high_mem.json
```
