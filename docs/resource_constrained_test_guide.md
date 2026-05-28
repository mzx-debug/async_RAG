# 资源受限场景测试指南 — V1 (BEIR 语料库)

> 本指南适配 V1 版本（资源受限技术栈 + BEIR 真实语料库）。
> **环境要求**: Python >= 3.10, CUDA 12.x, 4–16 GB VRAM GPU。

## 1. 环境配置

```bash
# 创建 conda 环境（推荐）
conda create -n async_rag python=3.10 -y
conda activate async_rag

# 安装 PyTorch CUDA 版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装 vLLM（关键依赖）
pip install vllm>=0.6.0

# 安装其余依赖
pip install -r requirements.txt

# 验证环境
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import vllm; print('vLLM:', vllm.__version__)"
```

> **中国内地**: 下载 HuggingFace 模型较慢，运行前设置 `export HF_ENDPOINT=https://hf-mirror.com`。
>
> **Python 3.13**: faiss-gpu 尚未发布，GPU retrieval (`--xR 1`) 在 Python 3.13 上不可用，只能用 `--xR 0`。

## 2. 技术栈概览

| 组件 | V0 (服务器) | V1 (受限设备) |
|------|------------|--------------|
| 生成模型 | Llama-3.1-8B-Instruct | Qwen2.5-1.5B-Instruct / 3B-Instruct |
| Embedding | e5-large-v2 (1024-dim) | all-MiniLM-L6-v2 (384-dim) |
| 语料库 | 880 万条（随机采样） | BEIR 真实语料库（3.6K–56K 条） |
| 索引 | IVF4096 (34 GB) | Flat / HNSW / IVF (~1–11 MB) |
| vLLM 显存占用 | 0.3 (模拟) | 0.6–0.8 (真实受限) |
| nprobe | 128 | 1 (Flat), >1 (IVF) |

## 2. BEIR 数据集选择

| 数据集 | 文档数 | 查询数 | 领域 | 索引大小 | 推荐显存 |
|--------|--------|--------|------|---------|---------|
| `nfcorpus` | ~3,600 | ~323 | 生物医学（饮食/健康） | ~1 MB | <= 6 GB |
| `scifact` | ~5,180 | ~1,109 | 科学声明检索 | ~1 MB | 6–12 GB |
| `arguana` | ~8,700 | ~1,409 | 论点检索 | ~2 MB | 12+ GB |
| `fiqa` | ~56,000 | ~6,600 | 金融 QA | ~11 MB | 大显存设备 |

所有数据集都包含真实的相关性标注（qrels），检索难度非随机采样可比。

## 3. 显存与模型选择（RTX 4090 Laptop 16GB 实测）

| 场景 | 模型 | `--gpu-memory-utilization` | `--xE` | `--xR` | 状态 |
|------|------|--------------------------|-------|-------|------|
| 推荐基线 | Qwen2.5-1.5B-Instruct | 0.80 | 0 | 0 | 稳定 |
| 3B 模型（CPU retrieval） | Qwen2.5-3B-Instruct | 0.60 | 0 | 0 | 稳定 |
| 3B 模型（GPU retrieval） | Qwen2.5-3B-Instruct | 0.45 | 0 | 1 | 显存紧张，可能 OOM |
| 1.5B + GPU retrieval | Qwen2.5-1.5B-Instruct | 0.80 | 0 | 1 | 可行但提升有限 |

> **注意**: GPU retrieval (`--xR 1`) 在 16GB 显存上搭配 Qwen2.5-3B-Instruct 会因显存不足导致 FAISS cublas GEMM 失败。async_v2 调度器会自动跳过不可行的 action，但 serial/async_plain 模式会直接崩溃。

## 4. 核心参数说明

| 参数 | 说明 | V1 默认值 |
|------|------|----------|
| `--gpu-memory-utilization` | vLLM 显存占用比例 | 0.6 |
| `--b` | CLI batch size | 32 |
| `--xE` | Embedding 设备: 0=CPU, 1=GPU | 0（对比时公平） |
| `--xR` | Retrieval 设备: 0=CPU, 1=GPU | 0（避免显存竞争） |
| `--nprobe` | FAISS probe 数 | 1 |
| `--enable-memory-aware-scheduling` | 显存感知调度开关 | 默认开启 |

## 5. Action 空间

```
xE=0, xR=0  → CPU embed + CPU retrieval  (释放最多 GPU 显存，**推荐**)
xE=1, xR=0  → GPU embed + CPU retrieval  (embedding 与生成争 GPU)
xE=0, xR=1  → CPU embed + GPU retrieval  (需要 faiss-gpu，显存允许时可用)
xE=1, xR=1  → GPU embed + GPU retrieval  (显存最紧张，仅 > 16GB 显存可行)
```

**V1 特有**: Flat 索引只有 ~1 MB，`xR=1` 在显存充足的设备上完全可行——这与 V0 的大索引场景完全不同。

## 6. 快速验证（推荐第一步）

用最小配置验证环境是否就绪：

```bash
# 只需 16 个查询，Qwen2.5-1.5B-Instruct
python ./async_rag_pipeline.py \
  --pipeline-mode serial \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --sample-queries 16 --b 4 \
  --xE 0 --xR 0 --nprobe 1 --topk 1 \
  --output-json ./output/test_serial.json

# 成功后再跑完整对比
python run_comparison.py --workdir . --sample-queries 256
```

## 7. 常见问题排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `No module named 'vllm'` | 未激活 conda 环境 | `conda activate <env>` |
| `torch.cuda.is_available()` = False | CUDA 版本不匹配 | 重新安装对应 CUDA 版本的 PyTorch |
| FAISS GPU OOM (cublas error 13) | vLLM 显存占用过高 | 降低 `--gpu-memory-utilization` 至 0.45，或用 1.5B 模型 |
| `Ninja build failed` (FlashInfer) | FlashInfer JIT 缓存损坏 | `rm -rf ~/.cache/flashinfer/` |
| 模型下载慢/失败 | 网络问题 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `Assertion 'err == CUBLAS_STATUS_SUCCESS'` | 同上，显存不足 | 同上，降低显存占用 |

## 8. 对照组实验命令

### 对照组 A: serial 基线

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode serial \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --b 32 --xE 0 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.80 \
  --output-json ./output/resource_test/serial.json
```

### 对照组 B: async_plain

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_plain \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --b 32 --xE 0 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.80 \
  --output-json ./output/resource_test/async_plain.json
```

## 9. 实验组：async_v2 调度器

### 实验组 A: EMA 在线调度（核心实验）

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_v2 \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --b 32 --xE 0 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.80 \
  --enable-memory-aware-scheduling \
  --output-json ./output/resource_test/async_v2.json
```

### 实验组 B: 测试 GPU retrieval 可行性

```bash
# 在显存充足的设备上测试 xR=1
python ./async_rag_pipeline.py \
  --pipeline-mode async_v2 \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --b 32 --xE 0 --xR 1 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.80 \
  --enable-memory-aware-scheduling \
  --output-json ./output/resource_test/async_v2_xR1.json
```

## 10. 消融实验

```bash
# 消融 A: 关闭显存感知调度
python ./async_rag_pipeline.py \
  --pipeline-mode async_v2 \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --b 32 --xE 0 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.80 \
  --disable-memory-aware-scheduling \
  --output-json ./output/resource_test/v2_no_mem_aware.json

# 消融 B: 固定 batch size
python ./async_rag_pipeline.py \
  --pipeline-mode async_v2 \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --b 32 --xE 0 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.80 \
  --enable-memory-aware-scheduling \
  --ablate-online-batch \
  --output-json ./output/resource_test/v2_fixed_batch.json

# 消融 C: 固定 action
python ./async_rag_pipeline.py \
  --pipeline-mode async_v2 \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --b 32 --xE 0 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.80 \
  --enable-memory-aware-scheduling \
  --ablate-online-action \
  --output-json ./output/resource_test/v2_fixed_action.json
```

## 11. 结果分析

### 6.1 关键指标

```python
wall_throughput_qps   # 实际吞吐量（最重要）
wall_time_ms          # 实际耗时
avg_emb_ms            # embedding 延迟
avg_ret_ms            # retrieval 延迟
avg_gen_ms            # generation 延迟
action_counts          # 各 action 被选中的次数
```

### 6.2 预期结论（V1 特有）

| 判断 | 标准 |
|------|------|
| **调度器在 V1 中仍然有效** | `async_v2` 的 QPS > `plain_b32` |
| **xR=1 在 V1 中有价值** | `action_counts` 中出现 `xR=1` 且 wall_time 更短 |
| **xE=0 策略有价值** | `action_counts` 中出现 `xE0` |
| **V1 vs V0 差异** | V1 中 retrieval 占比更高，xR=1 更常用 |
| **调度器无收益** | 所有 action 都是 `xE1_xR0`，等同于静态策略 |

## 7. V1 特有研究问题

V1 与 V0 最大的不同在于 **xR=1 的可行性**：

- V0: 34GB 索引，xR=1 几乎不可行
- V1: ~1MB 索引，xR=1 完全可行，且可能比 xR=0 更快

如果 async_v2 频繁选择 xR=1 并获得更好的 QPS，这说明在小索引场景下，GPU retrieval 确实有优势——这是 V0 无法验证的。

## 8. 显存感知参数调优

```bash
# 低显存 GPU (如 6GB 总显存)
python ./async_rag_pipeline.py \
  --pipeline-mode async_v2 \
  --gpu-memory-utilization 0.5 \
  --gpu-mem-low-threshold-gb 1.5 \
  --gpu-mem-medium-threshold-gb 3.0 \
  --gpu-mem-high-batch-penalty 30.0 \
  --output-json ./output/resource_test/tuned_low_mem.json

# 高显存 GPU (如 12GB 总显存)
python ./async_rag_pipeline.py \
  --pipeline-mode async_v2 \
  --gpu-memory-utilization 0.7 \
  --gpu-mem-low-threshold-gb 2.5 \
  --gpu-mem-medium-threshold-gb 6.0 \
  --output-json ./output/resource_test/tuned_high_mem.json
```
