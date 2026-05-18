# 研究进展记录

## 研究背景

基于初实验报告（RAG Batch Scaling），已完成：
- batch size 对 E/R/G 三阶段延迟的影响分析
- CPU vs GPU 后端对比
- Query 长度对各阶段延迟的影响

**初实验核心结论**：
- batch=256 是最优点（QPS 拐点）
- Generation 占主导（GPU 后端 batch=256 时占 69.5%）
- Retrieval 次之（29.9%），Embedding 可忽略（0.6%）
- 长 query 的瓶颈在 Embedding（CPU 后端 t256 时 428ms）
- 建议：Embedding 用 GPU，Retrieval 用 CPU

---

## 本阶段工作（async pipeline 实验）

### 研究问题
异步流水线（E/R/G 并发）能否提升 RAG 系统吞吐量？

### 实验设置（小库）
- 语料库：msmarco_2k.jsonl（2000 条）
- 索引：FAISS IVF256
- 模型：Llama-3.1-8B-Instruct（vLLM）+ e5-large-v2
- 问题集：queries_generated.jsonl（512 条，short/mid/long = 50%/35%/15%）
- 参数：b=16, xE=1, xR=0, nprobe=32, topk=1

### 小库实验结果

| 模式 | wall_time_ms | wall_QPS | avg_emb_ms | avg_ret_ms | avg_gen_ms |
|------|-------------|----------|-----------|-----------|-----------|
| serial | 84,998 | 6.02 | 1.97 | 0.56 | 163.47 |
| async_plain | 84,792 | 6.04 | 13.39 | 0.21 | 165.52 |
| async_bucket | 54,330 | 9.42 | 25.95 | 0.29 | 74.46 |

### 小库实验结论

1. **async_plain 无收益**：E+R 仅占 1.5%，流水线重叠空间不足；GPU 争抢导致 embedding 耗时 +580%
2. **async_bucket 快 36%**：原因是 batch size 从 16 涨到 62（short bucket=64），vLLM GPU 利用率提升，ms/token 从 1.28 降到 0.58
3. **真正的优化杠杆是 generation batch size，不是 E/R 调度**
4. **小库场景（retrieval 占 0.3%）不是 async pipeline 的适用场景**

---

## 大库实验（已完成，含修复后重跑）

### 实验设置
- 语料库：msmarco-passage-corpus（880 万条，Arrow 格式）
- 索引：ivf.index（34GB，IVF4096，来自 zenodo.org/records/16663591）
- 参数：b=16, xE=1, xR=0, nprobe=128, topk=1, gpu-id=3
- 问题集：queries_generated.jsonl（512 条）

### 大库实验结果（修复后，comparison_large）

| 模式 | wall_time_ms | wall_QPS | avg_emb_ms | avg_ret_ms | avg_gen_ms | ms/token |
|------|-------------|----------|-----------|-----------|-----------|---------|
| serial | 95,300 | 5.37 | 2.08 | 21.02 | 162.98 | 1.273 |
| async_plain | 85,744 | 5.97 | 12.15 | 14.42 | 166.80 | 1.303 |
| async_bucket | 37,523 | **13.64** | 1.11 | 17.31 | **70.74** | **0.553** |

### 各阶段占比（serial 基准）

```
Embedding:   2.08ms  ( 1.1%)
Retrieval:  21.02ms  (11.3%)
Generation: 162.98ms (87.6%)
```

### 大库实验结论

1. **Retrieval 占比从 0.3% 涨到 11.3%**，验证了"大库场景 retrieval 成为显著瓶颈"的预期
2. **async_plain 有了真实收益：+11%**（wall_time 从 95.3s 降到 85.7s），纯流水线重叠贡献
3. **async_bucket 快了 2.54×**（wall_QPS 从 5.37 涨到 13.64），= 流水线重叠 + batch size 效应叠加
4. **generation ms/token 从 1.27 降到 0.55**，与小库结论一致：batch size 是 generation 吞吐的主要杠杆
5. **async_bucket 的 avg_embedding_ms=1.11ms**（低于 serial 的 2.08ms），证实大 batch burst embedding 无 GPU 竞争
6. **action 空间开放后正确选择了 xE=1, xR=0**（GPU 可用显存不足 20 GiB，xR=1 被运行时过滤排除）

### 小库 vs 大库对比

| 指标 | 小库 (2K, nprobe=32) | 大库 (8.8M, nprobe=128) | 变化 |
|------|-----------|------------|------|
| retrieval 占比 | 0.3% | 11.3% | 38× 上升 |
| async_plain 收益 | 0% | +11% | 开始有价值 |
| async_bucket 收益 | +56% | +154% | 进一步扩大 |
| generation ms/token | 1.28 | 1.27 | 不变 |
| async_bucket ms/token | 0.58 | 0.55 | 不变 |

### 与论文对比（更新）

| 场景 | Retrieval 占比 | Async 收益 |
|------|--------------|-----------|
| 小库实验（2k 条，nprobe=32） | 0.3% | 0%（async_plain），+56%（async_bucket） |
| 大库实验（880 万条，nprobe=128） | 11.3% | +11%（async_plain），+154%（async_bucket） |
| 初实验报告（大库） | 29.9% | 未测试 |
| 论文场景（亿级） | 10-60% | 1.5-5× |

---

## 发现的 Bug 及修复

### Bug 1：调度器选择超出硬件约束的 action（致命）

**现象**：调度器给 short 桶选了 `xR=1`，retrieval_worker 试图把 34GB 索引加载到 GPU，CUDA 内存分配挂起。

**根因**：`available_actions` 包含所有 action，打分逻辑中 xR=1 得分更低（1.2 < 4.5），被选为最优 action，但 GPU 显存不足。

**修复历程**：
1. 第一版：用 CLI 参数过滤（`a["xE"] <= cli_xE and a["xR"] <= cli_xR`）
2. 当前版：开放完整 action 空间，改用运行时条件过滤（gpu_available、gpu_free_mem < 20 GiB、batch_size < threshold）

### Bug 2：short 桶被强制 CPU embedding（严重）

**现象**：short 桶 embedding 耗时 1400-2100ms/批（应该 60ms）。

**根因**：`_choose_action_for_batch` 的过滤条件 `if x_e == 1 and l_max < embed_mid_gpu_threshold(64)` 把 short 桶（l_max≤48）的 GPU embedding 全部过滤掉。

**修复**：删除该过滤条件。

### Bug 3：FAISS 默认单线程

**修复**：添加 `faiss.omp_set_num_threads()` + 边缘设备自动策略。

### Bug 4：大 batch retrieval 无中间日志

**修复**：超过 16 条的 batch 分片检索，每片打日志。

### Bug 5：GPU 显存检查用总显存而非可用显存

**现象**：开放 action 空间后，`gpu_mem_gb < 8.0` 检查用的是 GPU 总显存（47 GiB），vLLM 加载后实际可用仅 12 GiB，但 xR=1 未被过滤，触发 FAISS 索引重新加载到 GPU 导致 OOM/卡死。

**修复**：新增 `_detect_gpu_free_memory_gb()` 使用 `torch.cuda.mem_get_info()`，阈值从 8.0 提高到 20.0 GiB。

---

## 当前架构改进（已实现）

### 1. Action 空间开放
- 调度器拥有完整 {xE:0/1, xR:0/1} 空间，不再受 CLI 参数限制
- 运行时由 `_choose_action_for_batch` 根据 GPU 可用性/可用显存/batch 大小过滤不可行 action
- r=0.8 已从 action 空间移除，所有 query 以原始长度处理

### 2. 动态 Batch Size（延迟反馈）
- 每个桶独立跟踪 batch_size，基于 hill-climbing 逐 batch 调整
- 目标：最小化 per_query_latency = total_stage_time / batch_size
- 参数：step=8, min=8, max=128
- 背压机制作为安全阀（队列满时强制减半）

### 3. 切片嵌入仅限 async_bucket
- `chunked_embedding` 参数控制，仅 `pipeline_mode == "async_bucket"` 时启用
- serial/async_plain 使用简单 truncation，保证对比公平

### 4. Per-Batch Action 记录
- `BatchStats` 新增 xE/xR/r 字段
- 输出 JSON 中每个 batch 记录包含完整决策信息，便于分析算法行为

---

## 后续实验计划

1. **长 query 切分效果验证**：用 queries_long.jsonl 跑 async_bucket，验证切片嵌入的检索质量
2. **nprobe 对比实验**：nprobe=32/128/512，量化 retrieval 占比变化对 async 收益的影响
3. **在线延迟跟踪替代静态评分**：用 EMA 跟踪实际延迟，替代硬编码的 action 评分系数
4. **动态 batch_size 收敛验证**：用数千条 query 观察 hill-climbing 的收敛行为
5. **量化 async 适用边界**：绘制 retrieval 占比 vs async 加速比曲线
