# Async RAG Pipeline — 技术架构



---

## 一、流水线做什么

Async RAG Pipeline 是一个**异步检索增强生成系统**，核心目标是在 GPU 资源受限的环境下，通过动态调度 CPU 和 GPU 上的嵌入（Embedding）、检索（Retrieval）和生成（Generation）三个阶段，最小化每个查询的端到端延迟。

---

## 二、核心挑战：CPU 和 GPU 如何协同

RAG 推理有三个计算密集阶段：

```
查询 → [嵌入] → [检索] → [生成] → 回答
           ↓         ↓         ↓
          CPU或GPU  CPU或GPU    GPU
```

关键矛盾在于：**GPU 显存有限，不能同时跑太多任务**。具体来说，GPU 上需要承载：

- vLLM 生成模型的 KV Cache（占用 ~GPU总显存×80%）
- FAISS GPU 索引（~1.85 GB）
- 嵌入模型权重（~0.4 GB）

这意味着 GPU 无法同时高效处理嵌入、检索和生成三个任务。我们需要决定：

1. **嵌入在哪里跑**：CPU（慢但释放 GPU）还是 GPU（快但争抢资源）？
2. **检索在哪里跑**：CPU 还是 GPU？
3. **每个 batch 放多少查询**？

这就是调度器需要回答的问题。

---

## 三、三种执行模式


| 模式             | 调度方式                                       | 适用场景      |
| -------------- | ------------------------------------------ | --------- |
| `serial`       | 固定顺序，无并发                                   | 基准测试、标定实验 |
| `async_plain`  | 固定 action + 固定 batch size，三线程流水线           | 简单异步，不自适应 |
| `**async_v2`** | **在线贪心调度，每 batch 动态选 action 和 batch size** | **生产使用**  |


---

## 四、`async_v2` 的核心思想

```
┌─────────────────────────────────────────────────────────────┐
│  CPU 流水线:   [CPU嵌入+xfer_EtoR] → [CPU检索+xfer_RtoG]           │
│                   ↑ Emb 完成后跨设备传输   ↑ Ret 完成后传输给 Gen   │
│                                                             │
│  GPU 流水线:   [GPU嵌入+xfer_EtoR] → [GPU检索+xfer_RtoG] → [生成] │
│                                                             │
│  wall_q = max(emb + xfer_EtoR, ret + xfer_RtoG, gen) + queue_penalty │
│  xfer_EtoR = K[xE,xR] × L  (xE≠xR 时，加在 Emb)             │
│  xfer_RtoG = K[xR,1]  × L  (xR≠1  时，加在 Ret)             │
└─────────────────────────────────────────────────────────────┘
```

**为什么这样建模？**

- CPU 嵌入很慢（0.084 ms/token），但**不占用 GPU**，GPU 可以同时做生成
- GPU 嵌入很快（0.016 ms/token），但会和生成**抢 GPU 算力**
- 跨设备传输（xfer）不是独立阶段，而是增加发射端的工作完成代价
- 流水线并行运行，max 决定哪个工序成为瓶颈——流水线中较短的工作被吸收

---

## 五、成本模型（调度器的决策依据）

每做一个调度决策，调度器需要估算每种 `(xE, xR)` action 在给定 batch size 下的耗时。

### 5.1 模型公式

```
# 三阶段并行，xfer 加在前一阶段
wall_q = max(emb + xfer_EtoR, ret + xfer_RtoG, gen) + queue_penalty

其中：
  xfer_EtoR = I(xE≠xR) × K[xE,xR] × L   (Emb→Ret 传输，加在 Emb)
  xfer_RtoG = I(xR≠1)  × K[xR,1]  × L   (Ret→Gen 传输，加在 Ret)
  gen       = gen_per_token × avg_output + gen_base / B
  emb       = e[xE] × L + oh_e[xE] / B
  ret       = r[xR] + oh_r[xR] / B
```

**统一形式（推导后）：**

```
wall_q = max(emb + xfer_EtoR, ret + xfer_RtoG, gen) + queue_penalty
```

其中 `emb`、`ret`、`gen` 均为当前 action 下对应流水线侧的单 query 耗时，`xfer` 吸收到发射端后与 `emb` 合并计算。

### 5.2 参数含义


| 参数              | 默认值                | 来源           | 含义                      |
| --------------- | ------------------ | ------------ | ----------------------- |
| `e0`            | 0.36 ms/token      | 初始值          | CPU 嵌入速率                |
| `e1`            | 0.00 ms/token      | 初始值          | GPU 嵌入速率（overhead 主导）   |
| `oh_e0`         | 2.71 ms            | 初始值          | CPU 嵌入固定开销              |
| `oh_e1`         | 4.87 ms            | 初始值          | GPU 嵌入固定开销              |
| `r0`            | 0.20 ms            | 初始值          | CPU 检索边际成本              |
| `r1`            | 0.00 ms            | 初始值          | GPU 检索边际成本（overhead 主导） |
| `oh_r0`         | 1.36 ms            | 初始值          | CPU 检索固定开销              |
| `oh_r1`         | 1.50 ms            | 初始值          | GPU 检索固定开销              |
| `gen_per_token` | 0.2135 ms/token    | **Sweep 拟合** | 生成解码速率                  |
| `gen_base`      | 1109 ms            | **Sweep 拟合** | 生成预填充固定开销               |
| `avg_output`    | 120 tokens         | EMA 自适应      | 平均输出长度                  |
| `queue_penalty` | 0.23 ms/q          | Sweep 拟合     | 调度开销                    |
| `K[xE,xR]`      | 0.55/0.16 ms/token | 初始值          | Emb→Ret 传输速率（xE≠xR 时生效） |
| `K[xR,1]`       | 0.55 ms/token      | 初始值          | Ret→Gen 传输速率（xR≠1 时生效）  |


### 5.3 关键公式解读

**三阶段并行**（`max(emb+xfer_EtoR, ret+xfer_RtoG, gen)`）：Emb、Ret、Gen 三个阶段完全并行运行，三者的耗时加上各自的 xfer 后取最大值。xfer 不作为独立阶段——它加在前一个阶段的尾部，因为传输发生在该阶段完成后。

**xfer 加在发射端**：跨设备传输加在发射端（Emb 完成后给 Ret，Ret 完成后给 Gen），因为传输触发由发射侧驱动，它增加了发射侧工作的完成代价。

**统一 max 形式**：四种 action 共享同一结构 `wall_q = max(emb + xfer_EtoR, ret + xfer_RtoG, gen)`。当 gen 主导时（通常情况），公式退化为 `gen + queue_penalty`，即生成是唯一瓶颈；其余阶段在 gen 不主导时接管瓶颈。

**生成成本的分摊**（`gen_base / B`）：vLLM 的 continuous batching 下，每个 batch 的预填充（prefill）开销是固定的 1109 ms，与 batch size 无关。因此 batch 越大，每个查询分摊的预填充成本越低。这就是为什么更大的 batch 通常更快——但也不能无限大，因为 GPU 显存有限。

**检索的亚线性扩展**（`B^(α-1)`）：当 batch size 翻倍时，FAISS 检索时间不是翻倍，而是增长到 2^(α-1) 倍。α0=0.55 意味着 batch 翻倍，CPU 检索仅增加 47%；α1=0.30 意味着 GPU 检索仅增加 23%。这解释了为什么大 batch 在检索端很高效。

---

## 六、贪心调度器：如何选 action

每处理完一个 batch，调度器调用 `next_dispatch()`，在 **4 种 action × ~9 种 batch size = 36 种候选**中选耗时预测最短的一个：

```
候选生成：
  - batch_size ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}
  - action ∈ {(0,0), (0,1), (1,0), (1,1)}

筛选：
  1. GPU 显存够不够（ResourceTracker 检查）
  2. 下游队列是否积压（backpressure 触发放大 batch）

选择：
  → 预测耗时最短的 (action, batch_size) 组合
```

**显存感知**：ResourceTracker 实时监控 GPU 剩余显存（`torch.cuda.mem_get_info`），计算每个 action + batch size 的显存需求，在不够时跳过。**xR=1（GPU 检索）需要至少 ~2 GB FAISS GPU 索引驻留显存。**

**回压控制**：当下游队列（q_rg, q_out）积压超过阈值时，自动将 batch size 减半，防止系统过载。

---

## 七、三线程流水线架构

```
┌──────────────┐    q_er     ┌──────────────┐    q_rg     ┌──────────────┐
│ embed_worker │ ──────────→ │ retrieval_   │ ──────────→ │ generation_   │
│              │             │ worker       │             │ worker       │
│  (CPU 或 GPU) │             │  (CPU 或 GPU) │             │  (GPU/vLLM)  │
└──────────────┘             └──────────────┘             └──────────────┘
       ↑                                                            │
       │                 dispatch_cv                                │
       └─────────────────── main thread ────────────────────────────┘
                              ↑
                        scheduler.next_dispatch()
```

**工作流程**：

1. embed_worker 干完一个 batch → 通知主线程
2. 主线程调用调度器 → 决定下一个 action 和 batch size → 把任务塞进 q_er
3. retrieval_worker 从 q_er 取任务 → 干完塞进 q_rg
4. generation_worker 从 q_rg 取任务 → 干完记录反馈 → 通知主线程更新 EMA 参数

**关键设计**：embed_worker 每次干完都会停下来等主线程下一次调度。这种**半并发**模型确保了 action 可以随时切换，而不需要重启 worker 线程。

---

## 八、分块嵌入（Chunking）

当查询超过 64 token 时，系统采用滑动窗口分块策略：

```
查询（200 token）：|  0-63  |  48-111  |  96-159  |  144-207  |
                  chunk[0]    chunk[1]   chunk[2]   chunk[3]

所有查询的所有 chunk → 一次 forward → 各 chunk embedding → 平均 → 归一化
```

- 窗口大小：64 token
- 步长：48 token（重叠 16 token）
- **64 token 以下的查询：零开销**，直接处理
- 所有 chunk 合并成一个大 batch，一次模型推理搞定

**好处**：长查询（如金融文档摘要）不会被截断，语义更完整。

---

## 九、EMA 在线学习：边跑边适应

**每跑完一个 batch 都更新成本参数**：

```
new_value = 0.25 × observed + 0.75 × old_value
```

更新的内容：

1. **嵌入速率** `e[xE]`：从实际 embedding 耗时反推 per-token 速率
2. **检索参数** `r[xR]` 和 `α[xR]`：用滑动窗口数据拟合亚线性曲线
3. **生成速率** `gen_per_token`：从实际生成耗时计算 ms/token
4. **平均输出长度** `avg_output`：EMA 跟踪典型答案长度
5. **调度开销** `queue_penalty`：实测延迟减去预测延迟的残差

**warm-start**：每次运行结束保存所有参数，下次运行从上次状态继续，不需要重新标定。

---

## 十、完整工作流程图

```
用户查询流
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  GreedyScheduler.next_dispatch()                        │
│  • 遍历 4 种 action × 9 种 batch size                  │
│  • 用成本模型预测每种组合的 wall_time                   │
│  • ResourceTracker 检查 GPU 显存是否够                  │
│  • backpressure 检查队列是否积压                          │
│  • 选预测耗时最短的 (action, batch_size)               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Emb 阶段（并行）        │    Ret 阶段（并行）        │    Gen 阶段 │
│  [emb_cpu + xfer_EtoR] │    [ret_cpu + xfer_RtoG] │    [生成]     │
│  [emb_gpu + xfer_EtoR] │    [ret_gpu + xfer_RtoG] │              │
│                                                               │
│  wall_q = max(emb+xfer_EtoR, ret+xfer_RtoG, gen)               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  generation_worker 记录实际耗时                          │
│  → EMA 更新所有成本参数                                 │
│  → 保存 params JSON（可选）                             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
  返回结果
```

---

## 十一、当前验证状态

### 已验证（通过 Sweep 标定实验）


| 指标           | 结果                                     |
| ------------ | -------------------------------------- |
| Sweep 拟合质量   | `gen_ms = 1109 + 25.62 × B`，R² = 0.998 |
| (0,0) 预测误差   | < 3%（Sweep 条件下）                        |
| (0,1) 预测误差   | < 3%（Sweep 条件下）                        |
| prefill 分摊模型 | **完全正确**，gen_ms 随 B 反比变化符合预期           |
| 亚线性检索扩展      | **符合预期**                               |


### 已验证（通过 async_v2 真实场景验证）


| Scenario               | 误差        |
| ---------------------- | --------- |
| S1 (nfcorpus, gpu=0.8) | **-4.3%** |
| S5 (fiqa, gpu=0.8)     | **+8.6%** |
| S9 (nfcorpus, gpu=0.3) | **+0.1%** |


### 未验证（待补）


| 项目                           | 状态       | 说明                                          |
| ---------------------------- | -------- | ------------------------------------------- |
| (1,0) GPU 嵌入真实代价             | **待测**   | GPU 被 Ollama 占用，需释放后补跑                      |
| (1,1) GPU 检索真实代价             | **待测**   | 同上                                          |
| 连续 batch queue_penalty       | **待验证**  | 真实异步下 queueing 开销可能更大                       |
| 不同 gpu_util 下的 gen_per_token | **已知限制** | gpu=0.8 → 0.15 ms/tok，gpu=0.3 → 0.38 ms/tok |


---

## 十二、已知局限和待办

1. **(1,0)/(1,1) 参数未从数据拟合**：目前 GPU 端的 `e1`、`r1`、`α1` 全靠初始值，预测准确性未知。需要等 GPU 空闲后补跑。
2. **gen_per_token 对 gpu_util 敏感**：标定值 0.2135 在 gpu=0.8 时偏高（实际约 0.15），在 gpu=0.3 时偏低（实际约 0.38）。建议每个 gpu_util 单独标定。
3. **检索亚线性参数未拟合**：`r0, α0` 只在 (0,1) 部分数据上有约束，(1,1) 数据缺失导致 `r1, α1` 无法从数据中拟合。
4. **贪心策略可能局部最优**：36 候选中选最小预测耗时是贪心策略，没有 lookahead。在高度动态的负载下可能不如考虑未来状态的策略。

---

## 十三、关键文件索引


| 文件                        | 职责                                       |
| ------------------------- | ---------------------------------------- |
| `async_rag_pipeline.py`   | 全部逻辑：调度器、流水线线程、EMA、ResourceTracker       |
| `calibrate_sweep.py`      | 离线 Sweep：固定 action × 固定 batch 测真实延迟      |
| `compute_calib_params.py` | 从 Sweep 数据拟合 gen_base、gen_per_token      |
| `validate_model.py`       | async_v2 真实场景验证：预测 vs 实际对比               |
| `measure_gpu_costs.py`    | 固定 action 测 GPU embedding/retrieval 真实代价 |
| `改进文档.md`                 | 快速参考 + 最新实验数据                            |


