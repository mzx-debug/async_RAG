# 实验记录

---

## 实验 1：小库三模式对比（已完成）

**日期**：2026-05-12
**目的**：验证 async 流水线在小语料库场景下的收益

### 参数
```
语料库：msmarco_2k.jsonl（2000 条）
索引：indexes/ivf256_flat/faiss.index
模型：Llama-3.1-8B-Instruct + e5-large-v2
问题集：queries_generated.jsonl（512 条）
b=16, xE=1, xR=0, nprobe=32, topk=1, gpu-id=1
```

### 结果

| 模式 | wall_time_ms | wall_QPS | total_ms | avg_emb_ms | avg_ret_ms | avg_gen_ms | batches |
|------|-------------|----------|----------|-----------|-----------|-----------|---------|
| serial | 84,998 | 6.02 | 84,992 | 1.97 | 0.56 | 163.47 | 32×16 |
| async_plain | 84,792 | 6.04 | 91,708 | 13.39 | 0.21 | 165.52 | 32×16 |
| async_bucket | 54,330 | 9.42 | 51,556 | 25.95 | 0.29 | 74.46 | 7×62+3×26 |

**各阶段占比（serial）**：
- Embedding：1.97ms（1.2%）
- Retrieval：0.56ms（0.3%）
- Generation：163.47ms（98.5%）

### 关键发现
1. async_plain 无收益（wall_time 几乎持平）
   - 原因：E+R 仅占 1.5%，GPU 争抢导致 embedding +580%
   - 流水线确实工作（total_ms > wall_time，重叠节省 6.9s）
2. async_bucket 快 36%，原因是 batch size 从 16 涨到 62
   - ms/token：1.28 → 0.58（vLLM GPU 利用率提升）
   - 与三桶调度算法本身无关
3. 小库场景不适合 async pipeline（retrieval 占比太低）

### 结果文件
- `comparison/summary_serial.json`
- `comparison/summary_async_plain.json`
- `comparison/summary_async_bucket.json`
- `comparison/comparison_table.md`
- `comparison/analysis_report.md`

---

## 实验 2：大库三模式对比（已完成，含修复后重跑）

**日期**：2026-05-12
**目的**：验证大规模语料库场景下 async 流水线的收益

### 参数
```
语料库：data/msmarco-passage-corpus（880 万条，Arrow 格式）
索引：indexes/ivf_large/faiss.index（34GB，IVF4096）
模型：Llama-3.1-8B-Instruct + e5-large-v2
问题集：queries_generated.jsonl（512 条）
b=16, xE=1, xR=0, nprobe=128, topk=1, gpu-id=3
```

### 结果（修复后重跑）

| 模式 | wall_time_ms | wall_QPS | avg_emb_ms | avg_ret_ms | avg_gen_ms | ms/token |
|------|-------------|----------|-----------|-----------|-----------|---------|
| serial | 95,300 | 5.37 | 2.08 | 21.02 | 162.98 | 1.273 |
| async_plain | 85,744 | 5.97 | 12.15 | 14.42 | 166.80 | 1.303 |
| async_bucket | 37,523 | **13.64** | 1.11 | 17.31 | **70.74** | **0.553** |

**各阶段占比（serial）**：
- Embedding：2.08ms（1.1%）
- Retrieval：21.02ms（11.3%）
- Generation：162.98ms（87.6%）

### 关键发现
1. **async_bucket 加速 2.54×**（wall_QPS 5.37 → 13.64）
2. **generation ms/token 从 1.27 降到 0.55**：大 batch（61-63）充分利用 vLLM
3. **async_bucket embedding 无 GPU 竞争**：avg_emb=1.11ms（低于 serial 的 2.08ms），burst 模式有效
4. **async_plain GPU 竞争严重**：embedding 从 2ms 飙升到 12ms（batch 13-19 达 250-400ms）
5. **action 空间开放后正确选择 xE=1, xR=0**：GPU 可用显存 < 20 GiB，xR=1 被运行时过滤
6. **nprobe=128 时流水线重叠收益更大**：async_plain +11%，async_bucket +154%

### async_bucket 调度详情
```
bucket_counts: short=7, mid=3, long=0
action_counts: xE1_xR0_r1.0 = 10（全部选 GPU embedding + CPU retrieval）
batch sizes: short 桶 61-63, mid 桶 25-26
```

### 结果文件
- `comparison_large/summary_serial.json`
- `comparison_large/summary_async_plain.json`
- `comparison_large/summary_async_bucket.json`
- `comparison_large/comparison_table.md`

---

## 实验 3：长 Query 切分效果验证（待做）

**目的**：验证 64 token chunk 切分逻辑正常工作，embedding 耗时稳定

**注意**：切片嵌入现在仅在 async_bucket 模式下启用。serial/async_plain 使用简单 truncation。

### 参数
```
问题集：data/queries_long.jsonl（100 条，128-226 token）
b=16, xE=1, xR=0, pipeline-mode=async_bucket（验证切片）
对比：pipeline-mode=serial（无切片，truncation）
```

### 命令
```bash
# async_bucket（有切片嵌入）
python async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/ivf_large/faiss.index \
  --corpus-path ./data/msmarco-passage-corpus \
  --queries-file ./data/queries_long.jsonl \
  --sample-queries 100 \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-id 3 \
  --output-json ./output/long_query_bucket.json

# serial（无切片，truncation 对比）
python async_rag_pipeline.py \
  --pipeline-mode serial \
  --index-path ./indexes/ivf_large/faiss.index \
  --corpus-path ./data/msmarco-passage-corpus \
  --queries-file ./data/queries_long.jsonl \
  --sample-queries 100 \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --gpu-id 3 \
  --output-json ./output/long_query_serial.json
```

### 预期
- async_bucket：embedding 耗时稳定（切片后每 chunk 64 token，不暴涨）
- serial：embedding 更快（直接 truncation，无切片开销），但检索质量可能下降
- 对比两者的 answer 质量，验证切片嵌入的检索收益

---

## 实验 4：nprobe 对比（待做）

**目的**：量化 retrieval 占比随 nprobe 的变化，找到 async 收益的临界点

### 参数
```
nprobe = 32 / 128 / 512
其他参数固定：b=16, xE=1, xR=0, 大库
```

### 命令
```bash
# 只跑 serial 模式，快速获取各阶段占比
python async_rag_pipeline.py --pipeline-mode serial \
  --nprobe 32  --output-json ./output/nprobe32.json
python async_rag_pipeline.py --pipeline-mode serial \
  --nprobe 128 --output-json ./output/nprobe128.json
python async_rag_pipeline.py --pipeline-mode serial \
  --nprobe 512 --output-json ./output/nprobe512.json
```

---

## 实验 5：batch size 对比（待做）

**目的**：验证 PDF 报告结论（batch=256 最优），确认 async_bucket 的收益来源

### 参数
```
pipeline-mode = serial
b = 16 / 32 / 64
大库，nprobe=128
```

### 注意
- b=128 以上可能 OOM，不建议测试
- 重点观察 b=32 和 b=64 的 QPS 提升

---

## 实验 6：修复后 async_bucket 重跑（已完成）

**目的**：验证 Bug 修复 + 架构改进后 async_bucket 的真实性能

### 修复与改进内容
1. action 空间开放（运行时过滤替代 CLI 限制）
2. 删除 embed_mid_gpu_threshold 过滤（short 桶可用 GPU embedding）
3. FAISS 多线程 + 边缘设备策略
4. 大 batch 分片检索 + 日志
5. GPU 显存检查改用可用显存（阈值 20 GiB）
6. 动态 batch_size（hill-climbing 延迟反馈）
7. Per-batch action 记录（xE/xR/r）
8. 切片嵌入仅限 async_bucket

### 结果
见实验 2（修复后重跑）。

**验证**：
- [x] short 桶 GPU embedding ~50ms/批（从 2100ms 降到 50ms）
- [x] wall_time = 37.5s（预期 35-40s）
- [x] 不再卡死
- [x] action 正确选择 xE=1, xR=0（xR=1 被显存过滤排除）
- [x] per_batch 中可见 xE/xR/r 字段
