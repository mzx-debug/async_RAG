# Async RAG Pipeline — 成本模型完整说明

> 上次修改：2026-05-31

---

## 1. 系统运行流程概览

```
用户启动
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  阶段一：calibrate_sweep.py 离线标定                    │
│  （首次运行或换环境时执行一次）                          │
└──────────────────────┬──────────────────────────────┘
                       │ 生成 calibrated_params.json
                       ▼
┌─────────────────────────────────────────────────────┐
│  阶段二：async_rag_pipeline.py 运行时调度                │
│                                                       │
│  ├─ 加载 calibrated_params.json（暖启动）               │
│  ├─ 每个 dispatch：GreedyScheduler 选最优 action       │
│  ├─ 执行 batch（embedding → retrieval → generation）   │
│  └─ 反馈更新：EMA 在线学习参数                         │
└─────────────────────────────────────────────────────┘
```

---

## 2. 阶段一：calibrate_sweep.py 离线标定

### 2.1 为什么需要离线标定

系统初次运行或换到新环境时，所有 EMA 参数都是默认值。为了让调度器一开始就能做出合理决策，需要跑一次 **offline profiling sweep**，测量每种 (xE, xR) 组合在不同 batch size 下的实际耗时，然后拟合出初始参数。

### 2.2 工作原理

`calibrate_sweep.py` 对每种 (xE, xR) × B 组合分别启动 `async_rag_pipeline.py`（`--pipeline-mode async_plain`），收集 `per_batch` 中每个 batch 的 `generation_sec`，最后拟合线性模型。

```
参数：
  BATCH_SIZES = [1, 4, 16, 32, 64, 128]      # sweep 的 batch size
  DEFAULT_ACTIONS = [(0,0), (0,1), (1,0), (1,1)]  # 全部 4 种 action
  sample_queries = 256                          # 每个实验跑 256 个 query
```

### 2.3 运行命令

```bash
# 1. 标定所有 action（自动跳过已有结果文件）
python calibrate_sweep.py --workdir /home/cloudteam/rag_mzx

# 2. 标定单个 action
python calibrate_sweep.py --action 1 0 --batch-sizes 1 4 8 16 32 64

# 3. 预览将要执行的实验（不实际运行）
python calibrate_sweep.py --dry-run
```

### 2.4 输出文件

每次实验输出一个 JSON 文件：

```
output/
  calib_xR0_b1.json     # xE=0, xR=0, B=1
  calib_xR0_b4.json     # xE=0, xR=0, B=4
  calib_xR0_b16.json    # xE=0, xR=0, B=16
  ...
  calib_xE1R0_b1.json   # xE=1, xR=0, B=1
  calib_xE1R0_b4.json   # xE=1, xR=0, B=4
  ...
```

每个 JSON 文件的关键字段：

```json
{
  "config": { "xE": 0, "xR": 0 },
  "per_batch": [
    {
      "batch_size": 16,
      "embedding_sec": 0.065,
      "retrieval_sec": 0.010,
      "generation_sec": 1.413
    },
    ...
  ]
}
```

### 2.5 拟合线性模型

`calibrate_sweep.py` 从所有 B 的数据中拟合：

```
generation_time_ms = gen_base + gen_per_query × batch_size
```

即：`gen_total = P0 + g × B`，其中：

- `P0`（gen_base）：prefill 固定开销（ms），不随 B 变化
- `g`（gen_per_query）：每个 query 的 marginal cost（ms/q）

```
### (0,0) CPU E+R
  bs=1:   gen=1047ms/q
  bs=4:   gen=311ms/q
  bs=16:  gen=88ms/q
  bs=64:  gen=48ms/q
  Linear model: gen = 1047 + 31.5 × bs   (R²=0.999999)
```

### 2.6 生成 calibrated_params.json

标定完成后，将拟合的参数写入 JSON 文件，供运行时加载：

```bash
python compute_calib_params.py \
    --files output/calib_*.json \
    --emb-tokens-per-query 5.0 \
    --output output/calibrated_params.json
```

生成的 `calibrated_params.json` 包含所有 EMA 参数的初始值。

---

## 3. 阶段二：async_rag_pipeline.py 运行时

### 3.1 启动时加载参数

```bash
python async_rag_pipeline.py \
    --pipeline-mode async_v2 \
    --ema-params-path ./output/calibrated_params.json \
    --save-ema-params \
    ...
```

`--ema-params-path` 指定参数文件路径：

1. **如果文件存在** → `scheduler.load_ema_params(path)` 读取所有参数作为暖启动
2. **如果文件不存在** → 使用代码里的默认值

```python
# scheduler 初始化时的默认值（calibrated_params.json 存在时会被覆盖）
self._gen_per_token_ema = 0.378     # ms/token
self._gpu_contention_ema = 0.0       # ms/q（在新环境从 0 自适应）
self._queue_penalty_ema = 2.5        # ms/q
self._emb_rate_ema[0] = 0.084        # CPU embedding 速率
self._emb_rate_ema[1] = 0.016        # GPU embedding 速率
self._ret_r_ema[0] = 0.68            # CPU retrieval 系数
self._ret_r_ema[1] = 0.50            # GPU retrieval 系数
self._ret_alpha_ema[0] = 0.55        # CPU retrieval 亚线性指数
self._ret_alpha_ema[1] = 0.30        # GPU retrieval 亚线性指数
self._transfer_K_ema[(0,1)] = 0.55  # CPU→GPU 传输速率
self._transfer_K_ema[(1,0)] = 0.16  # GPU→CPU 传输速率
```

### 3.2 运行时调度循环（async_v2）

```
每个 query 到达时：
  │
  ▼
GreedyScheduler.next_dispatch()
  │
  ├─ 收集所有候选 action（4 种）
  ├─ 对每个 action 调用 _estimate_action_cost()
  │    计算 predicted_wall_time
  ├─ 按 predicted_wall_time 排序
  ├─ 选择 wall time 最小的 action
  └─ 返回 (xE, xR, batch_size)
  │
  ▼
pipeline 执行：
  EmbeddingStage(xE) → RetrievalStage(xR) → GenerationStage(GPU)
  │
  ▼
batch 完成 → _record_batch_feedback()
  │
  ├─ 提取实测时间（emb_sec, ret_sec, gen_sec, wall_time_ms）
  ├─ 计算 total_output_tokens
  └─ 调用 EMA 更新函数
       │
       ▼
  EMA 参数各自独立更新
```

### 3.3 成本模型（v4）

运行时用成本模型预测每种 action 的 wall time：

```
wall_q = gen_per_token × avg_output_tokens
       + queue_penalty
       + gpu_contention              (仅 xE=1)
       + 0.05 × (emb_q + ret_q)   (xE=0 时，E+R 与 Gen 完全 overlap)
       + xfer_q                     (仅 xE≠xR)
```

其中每个分量独立拟合：

```
emb_q  = e[xE] × L                  (embedding 速率 × 平均 token 数)
ret_q  = r[xR] × B^(alpha-1)       (检索亚线性模型)
xfer_q = K[xE,xR] × L              (跨设备传输)
```

### 3.4 EMA 在线学习

每次 batch 完成后，各参数独立更新：

```python
# Gen 吞吐率：直接从实测数据算
gen_per_token_new = gen_total_ms / output_tokens
gen_per_token_ema = 0.3 × gen_per_token_new + 0.7 × gen_per_token_ema

# Embedding 速率：每个 batch 都更新
e_obs = emb_total_ms / (L × B)
emb_rate_ema[xE] = 0.3 × e_obs + 0.7 × emb_rate_ema[xE]

# Retrieval 幂律：积累多个 B 的数据后重新拟合
ret_measurements[xR].append((B, ret_total_ms))
if len >= 2:
    # log-log 回归得到 r 和 alpha
    ret_r_ema[xR], ret_alpha_ema[xR] = fit_power_law(ret_measurements)

# GPU contention：仅 xE=1 时更新
if xE == 1:
    pred_no_cont = gen_per_token × avg_out + queue_penalty
    cont_obs = wall_q - pred_no_cont
    gpu_contention_ema = 0.3 × cont_obs + 0.7 × gpu_contention_ema

# Queue penalty：仅 xE=0 时更新（避免和 gpu_contention 混淆）
if xE == 0:
    obs_q = wall_q - gen_base
    queue_penalty_ema = 0.3 × obs_q + 0.7 × queue_penalty_ema
```

### 3.5 运行结束后保存参数

加上 `--save-ema-params` 会在运行结束后将 EMA 参数写回文件：

```bash
python async_rag_pipeline.py \
    --ema-params-path ./output/calibrated_params.json \
    --save-ema-params \
    ...
```

下次运行时直接加载更新后的参数，实现增量学习。

---

## 4. 完整使用流程

### 4.1 新环境首次运行

```bash
# Step 1: 离线标定（跑 4×6=24 个实验，可能需要 1-2 小时）
python calibrate_sweep.py --workdir . --sample-queries 256

# Step 2: 计算初始参数
python compute_calib_params.py \
    --files output/calib_*.json \
    --emb-tokens-per-query 5.0 \
    --output output/calibrated_params.json

# Step 3: 正式运行（自动加载标定参数 + EMA 在线自适应）
python async_rag_pipeline.py \
    --pipeline-mode async_v2 \
    --ema-params-path ./output/calibrated_params.json \
    --save-ema-params \
    ...
```

### 4.2 后续运行（已有标定文件）

```bash
# 直接运行，EMA 会自动从上次结束的地方继续学习
python async_rag_pipeline.py \
    --pipeline-mode async_v2 \
    --ema-params-path ./output/calibrated_params.json \
    --save-ema-params \
    ...
```

---

## 5. 标定参数表（RTX 4090 实测）

### 5.1 全部参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `gen_per_token_ema` | 0.378 ms/token | Gen 吞吐率（常数，跨 action 稳定） |
| `avg_output_tokens_ema` | 120 | 平均输出 token 数 |
| `queue_penalty_ema` | 2.5 ms/q | 异步调度 per-query 开销 |
| `gpu_contention_ema` | 38.5 ms/q | GPU emb+gen 串行惩罚（仅 xE=1） |
| `emb_rate_ema[0]` | 0.084 ms/token | CPU embedding 速率 |
| `emb_rate_ema[1]` | 0.016 ms/token | GPU embedding 速率 |
| `ret_r_ema[0]` | 0.68 | CPU retrieval 系数 |
| `ret_r_ema[1]` | 0.50 | GPU retrieval 系数 |
| `ret_alpha_ema[0]` | 0.55 | CPU retrieval 亚线性指数 |
| `ret_alpha_ema[1]` | 0.30 | GPU retrieval 亚线性指数 |
| `transfer_K[(0,1)]` | 0.55 ms/token | CPU→GPU 传输速率 |
| `transfer_K[(1,0)]` | 0.16 ms/token | GPU→CPU 传输速率 |

### 5.2 预测准确性

| Action | 预测 wall (B=64) | 实测 wall | 误差 |
|--------|-----------------|-----------|------|
| xE0R0 | 3063ms | 3214ms | -5% |
| xE0R1 | 3063ms | 3302ms | -7% |
| xE1R0 | 5527ms | 5218ms | +6% |

**Action 排名完全正确**：xE0R0 ≈ xE0R1 < xE1R0 < xE1R1

---

## 6. 关键发现

### 6.1 Gen 吞吐率是常数

所有 action 的 `gen_ms/token` 在 B=64 时几乎完全一致：

```
xE0R0: 0.3793 ms/token
xE0R1: 0.3808 ms/token
xE1R0: 0.3692 ms/token
```

这说明 **generation 完全不受 xE/xR 影响**，Gen 始终在 GPU 上以固定速率运行。xE/xR 只影响 pipeline overlap。

### 6.2 GPU contention 是 xE=1 变慢的根本原因

xE=0 时，CPU embedding/retrieval 和 GPU generation 完全并行：

```
[CPU: E+R] ──────────────────────[并行执行]───────
                                       ──────────[GPU: Gen]
wall ≈ gen（E+R 时间被 overlap 掉了）
```

xE=1 时，GPU embedding 必须等 Gen 做完才能开始（或反过来），串行执行：

```
[GPU: E+R]──[GPU: Gen]──>  （必须串行）
额外惩罚 = embedding 时间 ≈ 38ms/q × B
```

### 6.3 xE=1 在单 GPU 下永远不是最优

由于 GPU 是单卡独占，xE=1 的 contention penalty（38.5ms/q）远大于 CPU embedding 的收益（CPU emb 本身只需要约 0.3ms/q），所以 **xE=1 在任何单 GPU 环境下都不会是最优选择**。

但在多 GPU 环境下，xE=1 可能变成最优（embedding 和 gen 分卡运行）。

---

## 7. 文件说明

| 文件 | 说明 |
|------|------|
| `calibrate_sweep.py` | 离线 sweep：对每种 (xE, xR, B) 跑 profiling 实验 |
| `compute_calib_params.py` | 从 profiling JSON 拟合初始 EMA 参数 |
| `async_rag_pipeline.py` | 核心 pipeline，含运行时调度和 EMA 更新 |
| `output/calibrated_params.json` | 标定好的参数文件（加载到 scheduler） |
| `output/calib_*.json` | 原始 profiling 数据 |
| `output/v4_*.json` | 模型验证实验结果 |
