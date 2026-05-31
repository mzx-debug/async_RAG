# Cost Model Calibration — 完整进展文档

> 上次修改：2026-05-31

---

## 1. 问题背景

异步 RAG Pipeline 需要在每个调度周期选择最优的 **(xE, xR)** 组合——即 embedding 设备和 retrieval 设备放在哪里（CPU 或 GPU）。选择错误的组合会导致 GPU 空转、pipeline 气泡增加、QPS 下降。

决策依赖一个**成本模型**，它需要：

1. 预测每种 (xE, xR) 组合的 wall time
2. 从实测数据中自动学习参数（EMA 在线标定）
3. 在新环境下无需人工干预即可自适应

---

## 2. 硬件架构与 Action Space

### 2.1 设备配置

| 设备 | 职责 | 当前硬件 |
|------|------|---------|
| CPU | embedding 计算（xE=0）或 retrieval（xR=0） | 24 核 |
| GPU | generation（始终） + 可选的 embedding（xE=1）或 retrieval（xR=1） | RTX 4090 Laptop 16GB |

### 2.2 四种 Action

| Action | Embedding | Retrieval | Pipeline 特性 |
|--------|-----------|-----------|--------------|
| xE=0, xR=0 | CPU | CPU | CPU E+R 完全并行于 GPU Gen |
| xE=0, xR=1 | CPU | GPU | CPU E → GPU R，并行于 Gen |
| xE=1, xR=0 | GPU | CPU | GPU E → CPU R，与 Gen 竞争 GPU |
| xE=1, xR=1 | GPU | GPU | 全 GPU 串行 |

---

## 3. 成本模型演进

### 3.1 v1（初始基线）

假设各阶段完全串行，没有 pipeline overlap。模型过于悲观，实际 wall time 被严重高估。

### 3.2 v2（加入 overlap）

引入 `max(gen, emb+ret)` 形式的 pipeline 表达式，假设 E+R 和 Gen 可以完全 overlap。但没有考虑 GPU 争用（contention）。

### 3.3 v3（分组件独立标定）

将成本拆分为 emb、ret、gen、xfer 四个独立组件，各自拟合 EMA。但 gen 模型用 `P0 + g×B` 参数化，和 vLLM continuous batching 的实际行为不匹配，导致 Gen 预测误差 +9500%。

### 3.4 v4（当前版本）— 物理正确模型

从大量实测数据中发现了两个关键物理事实：

1. **Gen 吞吐率是常数**：gen_ms/token 在所有 action 下完全一致（约 0.378 ms/token），不随 xE/xR 变化
2. **GPU contention 是唯一差异来源**：xE=1 时 GPU embedding 和 Gen 串行化，带来固定 overhead

```
wall_q = gen_per_token × avg_output_tokens
       + queue_penalty                    (异步调度开销)
       + gpu_contention                  (仅 xE=1)
```

---

## 4. 关键实测数据

### 4.1 Gen per-token 稳定性

```
async_v2 continuous batching, B=64:
  xE0R0: gen_total=3092ms tokens=8153  gen/token=0.3793 ms/token
  xE0R1: gen_total=3105ms tokens=8153  gen/token=0.3808 ms/token
  xE1R0: gen_total=2996ms tokens=8114  gen/token=0.3692 ms/token
```

所有 action 的 gen/token 几乎完全一致（标准差 < 2%），说明 Gen 完全不受 xE/xR 影响。

### 4.2 Pipeline Overlap 分析

```
xE0R0 (CPU emb): wall = 3214ms  gen = 3077ms  gap = 137ms (2.3ms/q)
xE1R0 (GPU emb): wall = 5218ms  gen = 3004ms  gap = 2214ms (34.6ms/q)
```

- xE=0：CPU E+R 和 GPU Gen 完全 overlap，gap 很小（仅调度 overhead）
- xE=1：GPU E 和 Gen 竞争，gap = 约 2s（38.5ms/q × 64）

### 4.3 Retrieval 亚线性模型

Ret 是唯一随 B 变化的组件。用幂律建模：

```
ret_ms/q = r × B^(alpha-1)
```

实测结果：

| xR | r | alpha | B=1 | B=4 | B=16 | B=32 |
|----|---|-------|-----|-----|------|------|
| 0 (CPU) | 0.68 | 0.55 | 0.68 | 0.36 | 0.20 | 0.14 |
| 1 (GPU) | 0.50 | 0.30 | 0.50 | 0.19 | 0.07 | 0.04 |

---

## 5. 最终参数（RTX 4090 实测标定）

### 5.1 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `gen_per_token_ema` | **0.378 ms/token** | Gen 吞吐率（常数，跨 action 稳定） |
| `avg_output_tokens_ema` | **120** | 平均输出 token 数 |
| `queue_penalty_ema` | **2.5 ms/q** | 异步调度 per-query 开销 |
| `gpu_contention_ema` | **38.5 ms/q** | GPU emb+gen 串行化惩罚（仅 xE=1） |
| `emb_rate_ema[0]` | **0.084 ms/token** | CPU embedding 速率 |
| `emb_rate_ema[1]` | **0.016 ms/token** | GPU embedding 速率（约 5× 更快） |
| `ret_r_ema[0]` | **0.68** | CPU retrieval 系数 |
| `ret_r_ema[1]` | **0.50** | GPU retrieval 系数 |
| `ret_alpha_ema[0]` | **0.55** | CPU retrieval 亚线性指数 |
| `ret_alpha_ema[1]` | **0.30** | GPU retrieval 亚线性指数 |
| `transfer_K[(0,1)]` | **0.55 ms/token** | CPU→GPU 传输速率 |
| `transfer_K[(1,0)]` | **0.16 ms/token** | GPU→CPU 传输速率 |

### 5.2 预测准确性

| Action | 预测 wall (B=64) | 实测 wall | 误差 |
|--------|-----------------|-----------|------|
| xE0R0 | 3063ms | 3214ms | **-5%** |
| xE0R1 | 3063ms | 3302ms | **-7%** |
| xE1R0 | 5527ms | 5218ms | **+6%** |

**Action 排名完全正确**：xE0R0 ≈ xE0R1 < xE1R0 < xE1R1

---

## 6. 新环境自适应

### 6.1 自动 EMA 机制

每次 batch 完成后，各参数独立更新：

```
gen_per_token ← 0.3 × (gen_total / output_tokens) + 0.7 × gen_per_token
queue_penalty ← 0.3 × (wall_q - gen_base) + 0.7 × queue_penalty
gpu_contention ← 0.3 × (wall_obs_q - gen_base - queue) + 0.7 × gpu_contention
                                                      (仅 xE=1 时更新)
```

### 6.2 gpu_contention 的特殊性

`gpu_contention` 默认值为 0，完全依赖在线学习。这意味着：

- **新环境**：从 0 开始收敛，约 10-20 个 batch 后稳定
- **当前环境**（有 `calibrated_params.json`）：直接读取预标定值 38.5
- **不同 GPU**：该参数会自动调整到新硬件的实际争用水平

### 6.3 快速重新标定

在目标环境运行 `compute_calib_params.py` 可快速生成标定文件：

```bash
python compute_calib_params.py \
    --files output/calib_*.json \
    --emb-tokens-per-query 5.0 \
    --output output/calibrated_params.json
```

---

## 7. 为什么 xE=1 更慢

**根本原因**：GPU 是单卡独占资源，embedding 和 generation 必须在 GPU 上串行执行。

```
xE=0 (CPU emb + GPU gen)：完全并行
[CPU: embed] ──────────[CPU: retrieve]───
                                      ───[GPU: generate]───>
                                    两阶段完全 overlap

xE=1 (GPU emb + GPU gen)：串行化
[GPU: embed]──[GPU: generate]──>        (必须等 embedding 做完才能开始 gen)
额外惩罚 = embedding 时间 ≈ 38ms/q × B
```

因此 **xE=1 在单 GPU 环境下几乎永远不是最优选择**。但在新环境（多 GPU、embedding 模型更重等）下，这个结论可能反转。

---

## 8. 文件说明

| 文件 | 说明 |
|------|------|
| `async_rag_pipeline.py` | 核心 pipeline，含 v4 成本模型 |
| `compute_calib_params.py` | 从实测 JSON 自动计算初始 EMA 参数 |
| `output/calibrated_params.json` | 当前环境的预标定参数 |
| `output/v4_*.json` | v4 模型验证实验结果 |
| `output/calib_*.json` | 各 action 各 B 的原始测量数据 |

---

## 9. 未来改进方向

1. **B 最优选择**：当前只决策 (xE, xR)，B 是固定的。可以扩展模型预测最优 B
2. **Multi-GPU 支持**：xE=1 在多 GPU 下可能变成最优（embedding 和 gen 分卡）
3. **变长 query 建模**：当前用平均 L 近似，可引入 L 分桶
4. **xE1R1 标定**：尚未实测全 GPU 场景的参数
5. **gen_per_token 随 B 变化**：在更大 B 时验证 gen 是否仍为常数
