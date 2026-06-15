# 异构算力下面向多阶段 RAG 流水线的自适应调度研究

## 一、研究叙事线

```
┌──────────────────────────────────────────────────────────────────┐
│  核心矛盾:                                                         │
│  多阶段 RAG (Embedding → Retrieval → Generation) 在资源受限的      │
│  异构边缘设备 (CPU + GPU, 4-16 GB VRAM) 上运行。三个阶段必须共享     │
│  有限算力，但各自的资源需求特征截然不同——Embedding 是计算密集型，    │
│  Retrieval 是内存带宽密集型，Generation 是显存密集型（KV Cache）。    │
│  不存在一种"万能配置"能同时让三个阶段都高效运行。                      │
│                                                                    │
│  目标: 在异构算力约束下，动态调度最小化每个查询的端到端延迟             │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  Motivation: 测量驱动的系统分析 (四组实验)                           │
│                                                                    │
│  实验1: Batch Size 扩展规律                                        │
│    更大 batch 通过均摊固定开销降低延迟 (gen_base/B 从 1109→4.3ms)   │
│    → 调度器应倾向大 batch，但显存硬约束限制上限                       │
│                                                                    │
│  实验2: Query 长度影响规律                                          │
│    Embedding 对长度极度敏感 (34×)，Retrieval 近乎不敏感              │
│    长 query 需滑动窗口分块，实际计算量为 L_eff 而非 L                │
│    → 成本模型必须显式建模分块，用 L_eff 而非原始长度                  │
│                                                                    │
│  实验3: CPU vs GPU 加速比                                          │
│    Embedding GPU 加速比随 B 增长 (11×→146×)                        │
│    Retrieval GPU 加速比随 B 下降 (21×→10×)                         │
│    → 成本模型必须区分设备和阶段，逐 stage 建模                       │
│                                                                    │
│  实验4: 系统瓶颈迁移                                                │
│    GPU 下 Gen 主导 (70%)，CPU 下 Emb 可能反超                       │
│    瓶颈随 (action, B, L) 动态迁移                                   │
│    → 调度器必须在线决策，每 batch 重新评估                           │
└──────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼                                               ▼
┌─────────────────────────┐                 ┌──────────────────────┐
│  模块 1: 性能成本模型      │                 │  模块 2: 在线调度器    │
│                          │                 │                      │
│  回答"给定配置，多快？"    │    ──预测──→    │  回答"最优配置是什么？"  │
│                          │    ←──反馈──    │                      │
│  • 解析公式 (per-unit     │                 │  • 单调性分析 → B*    │
│    + 固定开销均摊)        │                 │    恒取可行域最大值    │
│  • L_eff 显式建模分块     │                 │  • 显存感知约束 B_mem  │
│  • 逐 stage 区分 CPU/GPU │                 │  • 4 action 比较 →    │
│  • EMA 在线参数估计      │                 │    最优设备配置        │
└─────────────────────────┘                 └──────────────────────┘
```

---

## 二、Introduction

### 2.1 背景

RAG（Retrieval-Augmented Generation）已成为提升 LLM 生成质量的关键技术。在隐私敏感和实时性要求高的场景下，RAG 推理正从云端向边缘设备迁移。然而，边缘设备的 GPU 资源极为有限（通常 4–16 GB VRAM），RAG 的三个计算密集阶段——**Embedding、Retrieval、Generation**——必须共享这些稀缺资源。

这三个阶段的计算特征截然不同：

| 阶段 | 计算特征 | 硬件偏好 | 显存占用 |
|------|---------|---------|---------|
| **Embedding** | 矩阵乘法密集，大 batch 下 GPU 加速比极高（~146×） | GPU 优先 | 模型权重 ~0.4 GB |
| **Retrieval** | 内存带宽密集，GPU 加速比有限且随 batch 增大下降 | CPU 也可 | FAISS 索引 ~1.85 GB |
| **Generation** | KV Cache 密集，必须 GPU（vLLM continuous batching） | GPU 独占 | KV Cache ~60-80% VRAM |

**核心矛盾**：vLLM 的 KV Cache 通常占据 60–80% 的显存，剩余空间不足以同时容纳 FAISS GPU 索引和 Embedding 模型权重。三个阶段必须通过**动态调度**在有限的异构算力上高效协同。

### 2.2 三阶段抽象的合理性

RAG 的工程变种繁多——稠密检索（DPR）、稀疏检索（BM25）、混合检索、多跳检索（Multi-hop）、Agentic RAG 等。但计算层面上，几乎所有变种共享一个最小公共结构：**将用户查询编码为检索表示 → 搜索知识库 → 将增强上下文输入 LLM 生成**。这三个阶段分别对应我们建模的 Embedding、Retrieval、Generation，是 RAG 推理延迟的主要来源。

我们的调度框架作用于这个**计算抽象层**，而非特定的 RAG 架构。额外的阶段（re-ranking、query rewriting 等）可以作为成本模型的扩展项加入，不影响调度方法本身。即使对于 AdaRAG 这样的多级检索架构，其 Light/Heavy retrieval 均可映射到我们的 Retrieval 阶段——区别仅在于参数的取值（`r`、`oh_r`）不同。

### 2.3 现有工作的不足

- **AdaRAG** 关注 retrieval 粒度和 prompt 设计的长周期在线决策，但**未涉及底层三阶段的设备级调度和 batch size 决策**
- **现有 RAG 系统**（如 vLLM + LangChain）通常采用固定的串行执行模式，未充分利用 CPU/GPU 异构算力的并行潜力
- **缺乏系统性的性能建模**：不同 query 长度、batch size、硬件配置下各阶段的延迟变化规律尚未被充分刻画

### 2.4 本文贡献

1. **实验驱动的系统分析**：通过 benchmark 实验，揭示 query 长度、batch size、设备配置三因素对各阶段延迟的异质性影响规律，以及瓶颈随条件动态迁移的现象
2. **异构性能成本模型**：针对 CPU/GPU 混合执行环境，建立三阶段延迟的轻量解析模型（~15 个参数），通过 EMA 在线学习跟踪硬件状态漂移，使模型在长周期运行中保持精度
3. **在线调度算法**：利用成本模型的单调性，将 batch size 优化退化为解析决策，在显存硬约束下实时选择最优设备配置

---

## 三、Motivation：实验驱动的系统分析

### 3.1 实验设置

| 组件 | 配置 |
|------|------|
| Embedding 模型 | intfloat/e5-large-v2（FP16, mean pooling） |
| 向量索引 | FAISS IVF4096 |
| 推理引擎 | vLLM (continuous batching) |
| 生成模型 | Llama3-8B |
| GPU | NVIDIA RTX 4090 |
| CPU | AMD EPYC 7K62 |
| 数据集 | NQ (Natural Questions) / MS MARCO |

### 3.2 发现 1：Batch Size 扩展规律（支撑显存感知调度）

**实验设计**：batch_size ∈ {8, 16, 32, 64, 128, 256, 512, 1024, 2048}，测量三阶段延迟变化，对比 CPU vs GPU 后端。

**关键观察**：

| batch_size | Embedding (ms) | Retrieval (ms) | Generation (ms) | Total (ms/query) | Throughput (QPS) |
|---|---|---|---|---|---|
| 8 | 3.24 | 20.35 | 277.27 | 300.85 | ~3.3 |
| 16 | 1.78 | 25.58 | 146.03 | 173.39 | ~5.8 |
| 32 | 0.97 | 17.85 | 84.02 | 102.83 | ~9.7 |
| 64 | 0.54 | 14.25 | 49.16 | 63.95 | ~15.6 |
| 128 | 0.32 | 12.32 | 33.33 | 45.98 | ~21.7 |
| 256 | 0.22 | 11.48 | 26.72 | 38.41 | ~26.0 |
| 512 | 0.25 | 10.59 | 26.68 | 37.52 | ~26.7 |
| 1024 | 0.22 | 10.82 | 26.73 | 37.77 | ~26.5 |

```
Embedding (GPU):
  bs=8 → 256: 延迟降低 ~93%
  bs>256: 趋于饱和

Retrieval (GPU):
  bs=8 → 256: 延迟降低 ~48%
  受 FAISS IVF 索引特性限制，加速比低于 Embedding

Generation (vLLM):
  bs=8 → 256: 延迟降低 ~90%
  gen_base / B 效应: 预填充固定开销 1109ms 在大 batch 下被有效均摊
```
 观察到随着batch指数增大，retrieval阶段下降近似线性，embedding与generation阶段在batch达到128前近似指数下降，可以认为在batch 16->32, 32->64阶段优化已经很有效。

> **Takeaway**: 更大的 batch 通常更快，但 batch 上限受 GPU 显存硬约束。需要**显存感知调度**——在运行时动态检查 GPU 显存，约束候选搜索空间。

---

### 3.3 发现 2：Query 长度影响规律（支撑输入感知调度）

**实验设计**：query token 长度分 6 档 {t8, t16, t32, t64, t128, t256}，batch_size 固定 128，测量各阶段延迟组成。

**关键观察**：

| 类别 | 实际 Token 均值 | Embedding (ms) | Retrieval (ms) | Generation (ms) | 吞吐量 (QPS) |
|------|----------------|---------------|---------------|----------------|-------------|
| t8   | 10.66          | **12.42**     | 15.18         | 84.96          | **8.88**    |
| t16  | 15.22          | **14.73**     | 11.82         | 87.92          | **8.74**    |
| t32  | 30.02          | **28.86**     | 12.99         | 90.89          | **7.53**    |
| t64  | 58.22          | **69.13**     | 14.08         | 102.34         | **5.39**    |
| t128 | 125.78         | **182.56**    | 13.01         | 110.22         | **3.27**    |
| t256 | 225.31         | **428.03**    | 20.87         | 129.48         | **1.73**    |

**核心洞察**：

- **Embedding 对 query 长度极度敏感**：t8 → t256 延迟增幅 ~34×（12.42ms → 428.03ms）
- **Retrieval 对 query 长度近乎不敏感**：可视为常数时间
- **Generation 对 query 长度近乎不敏感**
- **QPS 降幅达 80.5%**（8.88 → 1.73）

> **Takeaway**: Embedding 是长 query 的**主要瓶颈**。不同 query 长度对应不同的瓶颈阶段——短 query 受 Generation 主导，长 query 受 Embedding 控制。需要**输入感知的动态调度**。

---

### 3.4 发现 3：CPU vs GPU 加速比分析（支撑异构设备分配决策）

**实验设计**：对比 Embedding 和 Retrieval 在 CPU/GPU 上的加速比，覆盖 bs=8 到 bs=2048 全范围。

**关键观察**：

Embedding 对比 (CPU vs GPU)

| batch_size | CPU Embedding (ms) | GPU Embedding (ms) | GPU 加速比 |
|---|---|---|---|
| 8 | 24.35 | 2.14 | 11.4x |
| 32 | 17.19 | 0.67 | 25.7x |
| 128 | 20.95 | 0.41 | 51.1x |
| 512 | 28.43 | 0.23 | 123.6x |
| 2048 | 32.39 | 0.22 | 147.2x |

Retrieval 对比 (CPU vs GPU)

| batch_size | CPU Retrieval (ms) | GPU Retrieval (ms) | GPU 加速比 |
|---|---|---|---|
| 8 | 21.10 | 0.98 | 21.5x |
| 32 | 12.02 | 1.12 | 10.7x |
| 128 | 11.54 | 1.19 | 9.7x |
| 512 | 10.74 | 1.22 | 8.8x |
| 2048 | 12.02 | 1.21 | 9.9x |



**核心洞察**：

- Embedding 在大 batch 下 GPU 加速比远超 Retrieval
- Retrieval 的加速比不升反降，因为 FAISS 索引搜索受 CPU-GPU 数据传输带宽限制
- 资源受限时，应**优先将 GPU 分配给 Embedding 而非 Retrieval**

> **Takeaway**: 4 种设备配置 (xE, xR) ∈ {(CPU,CPU), (CPU,GPU), (GPU,CPU), (GPU,GPU)} 没有绝对最优，需根据当前 batch size 和 query 特征**动态选择**。



---

## 四、问题建模

### 4.1 系统模型

#### 调度粒度

| 维度 | 候选值 | 数量 |
|------|--------|------|
| **设备配置 (Action)** | (xE, xR) ∈ {(0,0), (0,1), (1,0), (1,1)} | 4 种 |
| **Batch Size** | 任意整数 B ∈ [1, min(pending_count, B_max(action))] | 动态变化 |

其中 xE=0 表示 Embedding 在 CPU，xE=1 表示在 GPU；xR 同理。

> **关于 batch size 候选空间**：batch size 不需要限定为 2 的幂次。任何整数 batch size 都可以被 vLLM continuous batching 支持。候选空间的大小在每个调度时刻动态确定：上界由 pending 队列长度 `pending_count` 和 GPU 显存约束 `B_max(action)` 共同决定，下界为 1。每个候选组合的评估仅涉及 ~10 次浮点运算，因此即使候选空间扩展到数百个，枚举开销仍然远低于调度本身的延迟（~0.23ms queue_penalty）。

#### 三阶段流水线架构

```
┌──────────────┐    q_er     ┌──────────────┐    q_rg     ┌──────────────┐
│ embed_worker │ ──────────→ │ retrieval_   │ ──────────→ │ generation_  │
│  (CPU 或 GPU) │             │ worker       │             │ worker       │
│              │             │  (CPU 或 GPU) │             │  (GPU/vLLM)  │
└──────────────┘             └──────────────┘             └──────────────┘
       ↑                                                            │
       │              dispatch_cv                                    │
       └─────────────────── main thread ────────────────────────────┘
                              ↑
                        scheduler.next_dispatch()
```

**关键设计**：embed_worker 每次处理完一个 batch 后等待主线程下一次调度。这种**半并发模型**确保设备配置可以随时切换，无需重启 worker。

#### 三阶段并行执行模型

三个阶段**完全并行运行**，端到端时间取最大值：

```
wall_q = max(
    emb_q + xfer_EtoR,
    ret_q + xfer_RtoG,
    gen_q
) + queue_penalty
```

其中：

queue_penalty：调度器计算与调度的固定开销

数据传输惩罚：

- `xfer_EtoR = I(xE ≠ xR) × K[xE, xR] × L`：当 Embedding 和 Retrieval 在不同设备时触发
- `xfer_RtoG = I(xR ≠ 1) × K[xR, 1] × L`：当 Retrieval 不在 GPU 时触发

### 4.2 优化问题形式化

```
目标: 最小化每个查询的端到端延迟 wall_q

决策变量:
  - 设备配置: action = (xE, xR) ∈ {(0,0), (0,1), (1,0), (1,1)}
  - Batch Size: B ∈ [1, min(pending_count, B_max(action))], B 为整数

约束:
  - GPU 显存约束: mem_required(action, B) ≤ gpu_available
  - 批大小约束: B ≤ pending_count（不超过待处理查询数）

在每次调度时刻 t:
  (action*, B*) = argmin_{action ∈ Actions, B ∈ [1, B_feasible]} wall_q(action, B | L, H_t)
  subject to 上述约束

注: B 是连续整数空间，每个 action 的候选数 = min(pending_count, B_max(action))，
    总候选数 = Σ_{action} min(pending_count, B_max(action))，通常为数十到数百个。
    每个候选仅需 ~10 次浮点运算评估，枚举开销远低于调度本身的延迟（~0.23ms）。
```

其中：
- `L`：当前 batch 中 query 的平均长度
- `H_t`：t 时刻的系统硬件状态（GPU 利用率、温度等）
- `B_max(action)`：给定设备配置下 GPU 显存能支持的最大 batch size

---

## 五、系统设计

### 5.1 架构总览

系统由两个模块构成：

```
┌──────────────────────────────────────────────────────────────┐
│                     Adaptive Scheduler                        │
│                                                               │
│  ┌──────────────────────────────┐  ┌──────────────────────┐  │
│  │ 模块 1: 性能成本模型            │  │ 模块 2: 在线调度器      │  │
│  │                              │  │                      │  │
│  │  ┌──────────────────────┐    │  │  ┌──────────────┐    │  │
│  │  │ 解析延迟公式           │    │  │ │ 显存感知约束   │    │  │
│  │  │ (per-unit + 均摊)     │    │  │ │ (B_max计算)   │    │  │
│  │  └──────────────────────┘    │  │ └──────────────┘    │  │
│  │            │                 │  │                      │  │
│  │  ┌──────────────────────┐    │  │  ┌──────────────┐    │  │
│  │  │ EMA 在线参数估计       │◄───┼──│ │ 解析B*决策    │    │  │
│  │  │ (分步拟合 / 指数加权)   │    │  │ │ (4 action比较)│    │  │
│  │  └──────────────────────┘    │  │ └──────────────┘    │  │
│  │                              │  │                      │  │
│  └──────────────────────────────┘  └──────────────────────┘  │
│         │                ▲                   │                │
│         │  预测 wall_q    │  实测反馈          │  输出          │
│         └────────────────┼───────────────────┘  (action, B)   │
│                          │                                      │
└──────────────────────────────────────────────────────────────┘
```

- **模块 1（成本模型）**回答"给定配置 (action, B, L)，端到端延迟是多少？"。包含解析公式和 EMA 在线参数估计，后者利用运行时反馈维持模型精度
- **模块 2（调度器）**回答"当前最优配置是什么？"。利用模型的单调性将 batch size 决策退化为解析推导，在显存硬约束下比较 4 种设备配置

---

### 5.2 模块 1：异构性能成本模型

#### 模型结构

将各阶段的执行时间分解为**per-unit 边际成本 + 固定开销均摊**：

#### Embedding 阶段

```
L_eff = num_chunks × chunk_size
      = ceil(max(0, L - overlap) / step) × chunk_size

emb_q = e[xE] × L_eff + oh_e[xE] / B
```

| 参数 | 含义 | 单位 |
|------|------|------|
| `e[xE]` | 单位 token 嵌入成本（取决于设备 xE） | ms/token |
| `oh_e[xE]` | 嵌入固定开销（启动延迟、CUDA 同步等） | ms |
| `L` | query 的原始 token 长度 | tokens |
| `L_eff` | 分块后的有效计算长度 | tokens |
| `B` | batch size | — |

**长 query 的分块处理**：Embedding 模型有最大输入长度限制（如 64 tokens）。超过此限制的 query 采用滑动窗口分块——`chunk_size=64, overlap=32, step=32`——所有 chunk 在一次 forward pass 中嵌入，同 query 的多个 chunk embedding 取平均后归一化。因此 200 token 的 query 实际被切分为 6 个 chunk（共 384 token 的计算量），而非原始长度的 200 token。

成本模型中使用 `L_eff` 而非 `L`，将分块开销显式纳入。这确保了成本模型对所有 query 长度的预测精度，尤其是在长 query 场景下不会显著低估 Embedding 时间。

#### Retrieval 阶段

```
ret_q = r[xR] + oh_r[xR] / B
```

| 参数 | 含义 | 单位 |
|------|------|------|
| `r[xR]` | 单查询检索边际成本（FAISS O(1) flat search 近似） | ms/q |
| `oh_r[xR]` | 检索固定开销（索引加载/初始化均摊） | ms |

**物理含义**：FAISS 检索时间近似 per-query 固定成本，与 query 长度无关（验证于发现 2）。

#### Generation 阶段

```
gen_q = gen_per_token × avg_output_tokens + gen_base / B
```

| 参数 | 含义 | 单位 |
|------|------|------|
| `gen_per_token` | 每 token 解码时间 | ms/token |
| `gen_base` | 预填充（prefill）固定开销 | ms |
| `avg_output_tokens` | 平均输出 token 数（EMA 自适应跟踪） | tokens |

**物理含义**：vLLM continuous batching 下，预填充开销与 batch size 无关，在大 batch 中被有效均摊（验证于发现 1）。

#### 完整预测模型

```
wall_q = max(
    e[xE] × L_eff + oh_e[xE] / B + I(xE≠xR) × K[xE,xR] × L_eff,  # Embedding + 传输
    r[xR] + oh_r[xR] / B + I(xR≠1) × K[xR,1] × L_eff,            # Retrieval + 传输
    gen_per_token × avg_output + gen_base / B                      # Generation
) + queue_penalty
```

#### 默认参数
### 7.0 实验配置

为了确保资源受限情景，调度器的实际运行在边缘设备上进行，并依据设备实际情况选用了更小规模的语料库、推理模型与嵌入模型。
| 组件 | 配置 |
|------|------|
| **Embedding 模型** | sentence-transformers/all-MiniLM-L6-v2（FP16, mean pooling，384-dim） |
| **向量索引** | FAISS Flat |
| **推理引擎** | vLLM（continuous batching） |
| **生成模型** | Qwen/Qwen2.5-1.5B-Instruct |
| **GPU** | NVIDIA RTX 4090（24 GB） |
| **CPU** | Intel(R) Core(TM) i9-14900HX |
| **数据集** | BEIR-nfcorpus（queries + corpus） |
| **vLLM gpu-memory-utilization** | 0.80 |

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `e0` (CPU emb) | 0.36 ms/token | CPU embedding rate |
| `e1` (GPU emb) | ≈0.00 ms/token | GPU 下 token 计算极快，时间几乎全为固定开销 |
| `oh_e0` (CPU emb 固定) | 2.71 ms | CPU embedding 启动延迟 |
| `oh_e1` (GPU emb 固定) | 4.87 ms | GPU embedding 启动延迟（比 CPU 更高） |
| `r0` (CPU ret) | 0.20 ms/q | 固定 FAISS 索引 O(1) 近似 |
| `r1` (GPU ret) | ≈0.00 ms/q | GPU 下检索极快 |
| `oh_r0` (CPU ret 固定) | 1.36 ms | CPU 索引加载开销均摊 |
| `oh_r1` (GPU ret 固定) | 1.50 ms | GPU 索引加载开销均摊 |
| `gen_per_token` | 0.2135 ms/token | **实测** |
| `gen_base` | 1109 ms | **实测** prefill 固定开销 |
| `avg_output` | 120 tokens | EMA 自适应 |
| `queue_penalty` | 0.23 ms/q | **实测** |
| `K[(0,1)]` (CPU→GPU) | 0.55 ms/token | 传输速率 |
| `K[(1,0)]` (GPU→CPU) | 0.16 ms/token | 传输速率 |

#### 5.2.4 EMA 在线参数估计

成本模型参数（嵌入速率、检索成本、生成速率）随硬件温度和 GPU 利用率动态变化。离线标定只能给初始值。每个 batch 执行完毕后，利用实测的延迟数据对参数进行 EMA 更新，使模型在长周期运行中保持预测精度。

**EMA 公式**：

```
new_value = α × observed + (1 - α) × old_value,  α = 0.25
```

**关键设计：预更新值冻结**。所有 EMA 更新前，先将当前值复制一份作为"预更新值"，再进行所有写入。这避免了同一 batch 内参数之间的循环引用——例如 emb 的残差影响 ret 的估计，ret 的残差又影响 gen 的估计。

**分步拟合流程**：

```
Step 1 — Embedding 拟合:
  收集 (L, B, emb_ms_total) 数据点
  模型: emb_total = e_base × (L × B) + overhead
  方法: 指数加权线性回归，窗口 ≤ 20 个数据点

Step 2 — Retrieval 拟合:
  模型: ret_total = r_base × B + overhead
  对称于 Embedding 拟合

Step 3 — Generation 拟合:
  per-token rate: gpt_obs = gen_ms_total / total_output_tokens
  gen_base: 从总时间剥离 per-token 成本后反解
  短输出 (<200 token) 的 batch 对 gen_base 更新权重较小

Step 4 — 传输 / Queue 拟合:
  从 wall_time 残差反推
  仅在慢阶段不是 gen 时才更新传输速率 K
```

---

### 5.3 模块 2：在线调度器

#### 核心观察：wall_q 对 B 的单调性

对于固定的设备配置 action = (xE, xR)，成本模型中三项均具有 `a + c / B` 的形式：

```
emb(B) = e[xE] × L + oh_e[xE] / B + xfer_EtoR    (xfer 与 B 无关)
ret(B) = r[xR] + oh_r[xR] / B + xfer_RtoG
gen(B) = gen_per_token × avg_output + gen_base / B
```

三个分量都是 B 的单调递减函数。**单调递减函数族的 pointwise maximum 仍然是单调递减的**：

```
∂wall_q/∂B < 0,  ∀B > 0
```

因此对于任意给定的 action，最优 batch size 就是**最大可行 batch size**：

```
B*(action) = min(pending_count, B_mem(action), B_backpressure)
```

这消除了对 batch size 维度的搜索——调度问题从"4×N 候选枚举"退化为"4 个 action 的比较"。

#### 算法流程

```
输入: pending_count, gpu_mem_gb, q_rg_len
输出: ScheduledMicrobatch

1. 如果没有 pending queries → 返回 None

2. FOR EACH action ∈ {(0,0), (0,1), (1,0), (1,1)}:
     a. B_mem = ResourceTracker.max_batch_size_for_action(xE, xR)
     b. B* = min(pending_count, B_mem)
     c. If B* < 1: skip this action
     d. wall_q = cost_model.estimate(action, B*)

3. 选择 wall_q 最小的 action，以其对应的 B* 打包 batch
```

#### 为什么 B* 总是取最大值

直觉上，更大的 batch 通过均摊固定开销降低了每个阶段的时间。因为所有阶段都受益（Embedding 的 `oh_e/B`、Retrieval 的 `oh_r/B`、Generation 的 `gen_base/B`），wall_q 中的三个分量同时下降，瓶颈阶段的时间必然下降。因此不需要遍历 B。

唯一的例外是当 batch 增大触发 GPU OOM 时——但这被 `B_mem(action)` 硬约束排除在外。

#### 伪代码

```python
def next_dispatch(self, pending_count, gpu_mem_gb):
    if pending_count == 0:
        return None

    best_action = None
    best_B = 0
    best_wall_q = float('inf')

    for xE, xR in [(0,0), (0,1), (1,0), (1,1)]:
        B_mem = self.resource_tracker.max_batch_size_for_action(xE, xR)
        B = min(pending_count, B_mem)
        
        if B < 1:
            continue
        
        wall_q = self._estimate_wall_time(B, xE, xR)
        if wall_q < best_wall_q:
            best_wall_q = wall_q
            best_action = (xE, xR)
            best_B = B

    return self._pack_batch(best_B, best_action)
```

#### 5.3.1 显存感知约束

每个 action 的最大可行 batch size 由 GPU 显存硬约束决定：

```python
def max_batch_size_for_action(self, xE: int, xR: int) -> int:
    required = self._estimate_model_mem(xE, xR)   # 该 action 下各模型/索引的显存需求
    available = (torch.cuda.get_device_properties(0).total_memory
                 - self._used_mem)
    return int(available / required)
```

`B_mem(action)` 直接作为 B* 的上界进入调度决策：`B* = min(pending_count, B_mem(action))`。不需要显存压力分级表——单调性保证了 B* 就是最优的，显存约束只是限制了可行域的上界。

此外，当选择 (xE=1, xR=1) 且 GPU 可用时，FAISS GPU 索引保持在显存中（`keep_gpu_resident_er`），避免每次重新加载的 ~300ms 开销。

---

## 六、实验验证计划

### 6.1 离线标定验证

| 指标 | 方法 | 预期结果 |
|------|------|---------|
| 生成线性模型 | `gen_ms = gen_base + gen_per_token × B` 拟合 | R² > 0.99 |
| 各配置预测误差 | 在 (0,0), (0,1), (1,0), (1,1) 下对比预测 vs 实测 | < 5% |
| 预填充均摊模型 | 验证 gen_ms 随 B 反比变化 | 严格符合理论 |

### 6.2 真实流水线验证

| 场景 | 数据集 | GPU 利用率 | 预期误差 |
|------|--------|-----------|---------|
| S1 | NQ | 0.8 | < 10% |
| S2 | MS MARCO | 0.8 | < 10% |
| S3 | NQ | 0.3（模拟低负载） | < 10% |
| S4 | MS MARCO | 0.3 | < 10% |

### 6.3 端到端性能对比

| 对比基线 | 说明 | 预期加速比 |
|---------|------|-----------|
| Serial B=1 | 串行执行，batch=1 | **~20×** |
| Serial B=8 | 串行执行，batch=8 | **~3×** |
| Serial B=32 | 串行执行，batch=32 | **~1.5×** |
| Random Scheduler | 随机选择设备配置和 batch | **~2×** |
| Static Best | 离线确定的最优固定配置 | **~1.3×** |

### 6.4 消融实验

| 消融模块 | 说明 | 预期影响 |
|---------|------|---------|
| 无 EMA | 仅使用默认参数，无在线学习 | 长周期运行中误差逐渐增大，调度决策偏离最优 |
| 无显存感知 | 不考虑 GPU 显存约束，仅按 pending_count 取 B_max | OOM 错误 |
| 无回压控制 | 不限制下游队列积压 | 系统过载，延迟飙升 |
| 固定 Batch | 始终使用固定 batch size（如 B=32） | 长 query 性能大幅下降 |
| 固定 Action | 始终使用同一设备配置（如 GPU-GPU） | 特定 query 分布下性能倒退 |

---

## 七、与 AdaRAG 的关系与差异化

| 维度 | AdaRAG | 本工作 |
|------|--------|--------|
| **优化层次** | 上层策略：retrieval 粒度 + prompt 设计 | 底层执行：设备调度 + batch size |
| **优化方法** | Bandit 凸优化（理论保证） | 解析调度 + EMA 在线学习（工程实用） |
| **时间尺度** | 长周期（slot-by-slot，跨越多个 batch） | 短周期（batch-by-batch，每次调度） |
| **资源建模** | CPU/GPU 流水线重叠（宏观） | GPU 显存硬约束 + 三阶段微观测算 |
| **性能模型** | 解析近似（凸性假设） | 轻量解析模型 + EMA 在线校准 |
| **硬件感知** | 未深入 | 深度建模（10+ 参数覆盖三阶段） |
| **互补性** | **上层策略输出**：p_t (heavy 比例) | **底层执行输入**：p_t 已知，决定如何调度执行 |

> 二者构成完整的 **Edge RAG 分层调度架构**：AdaRAG 决定"检索到什么程度"（战略层），本工作决定"用什么硬件、多大 batch 来执行"（战术层）。

---

## 八、论文大纲（草稿）

```
1. Introduction
   1.1 Edge RAG 的背景与挑战
   1.2 三阶段抽象的合理性
   1.3 现有工作的不足
   1.4 本文贡献

2. Motivation: 测量驱动的系统分析
   2.1 Batch Size 扩展规律实验 → 显存感知需求
   2.2 Query 长度影响规律实验 → 输入感知需求
   2.3 CPU vs GPU 加速比分析 → 异构调度需求
   2.4 系统瓶颈迁移分析 → 动态调度需求
   2.5 设计需求总结

3. System Model and Problem Formulation
   3.1 系统架构与三阶段流水线
   3.2 异构并行执行模型
   3.3 调度问题形式化

4. System Design
   4.1 性能成本模型
       4.1.1 解析延迟公式（三阶段分解）
       4.1.2 EMA 在线参数估计
   4.2 在线调度器
       4.2.1 单调性分析与解析 B* 决策
       4.2.2 显存感知约束

5. Implementation
   5.1 三线程异步流水线
   5.2 参数离线标定工具

6. Evaluation
   6.1 实验设置
   6.2 离线标定质量
   6.3 端到端性能对比
   6.4 消融实验
   6.5 在线学习收敛性分析

7. Related Work

8. Conclusion
```

---

## 九、后续需要补充的内容

1. **Batch 内长度异质性**：成本模型对 Embedding 使用 batch 内平均的 `L_eff`，对长度方差大的 batch 误差较大。可引入方差修正项，或按长度分组 batching 减少异质性

2. **更丰富的实验数据集**：除 NQ 外，需要覆盖更多 domain（如 HotpotQA、TriviaQA、MS MARCO）

3. **与 AdaRAG 的联合实验**：在上层 AdaRAG 决策下，验证底层调度器的实际增益

4. **多 GPU / 边缘集群扩展**：当前仅考虑单 GPU 场景，可扩展到多卡并行或边缘集群下的分布式调度
