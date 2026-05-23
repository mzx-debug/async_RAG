# Async RAG Pipeline 进展总结（2026-05-16）

---

## 0. 2026-05-23 更新：研究方向切换到资源受限场景

### 为什么现在要切换场景

已有实验在 GPU 显存充裕的服务器环境下运行。此时：
- `xE=1, xR=0, b=64` 是固定最优解，调度器几乎无优化空间
- `async_bucket` 无法显著超过 `plain_b64`
- `generation_target_v1` 方向暂停（未验证）

### 核心假设

在**资源受限**场景（GPU 显存不足、FAISS index 无法全驻留 GPU、vLLM 被限制在低 `gpu_memory_utilization`），`xE/xR/batch_size` 的联合决策才真正有意义：

| 充裕场景 | 资源受限场景 |
|---------|------------|
| xE=1, xR=0 是唯一最优 | xE/xR/batch_size 需权衡 |
| 大 batch 是免费的 | 大 batch 受显存约束 |
| CPU embed 无意义 | CPU embed 释放 GPU 给 retrieval+gen |
| Bucketing 净损失 | Bucketing 可通过 overlap 获益 |

### 已实现的改动

1. `ResourceTracker` — GPU 显存实时监控 + 各阶段预估
2. `_action_feasible()` — 显存感知 action 选择（替代硬编码 20 GiB 阈值）
3. `_bucket_priority()` — 显存压力驱动的桶优先级
4. `lookahead dispatch` — 真正的流水线 overlap
5. 新增 6 个 CLI 参数控制上述行为

### 下一步

**先在资源受限场景下重新跑三模式对比**，验证显存感知调度是否打开了新的优化空间。

---



## 1. 工作范围

本文档用于总结 `async_rag_pipeline_v0` 到当前为止的代码与实验进展。

本阶段工作主要覆盖四个方面：

1. 清理并收缩 scheduler 接口
2. 将 `async_bucket` 重构为在线 dispatch 工作流
3. 为 `summary_*.json` 增加详细可观测性
4. 在普通问题集与超长问题集上运行 comparison / ablation 实验并解释结果

本文档是当前阶段的高层参考，不是完整 changelog。

---

## 2. 主要代码演进

### 2.1 参数清理

原脚本中存在一批“名义可配、实际不影响运行”的参数。  
这些参数已经被删除或弱化，使 CLI、代码路径和 summary 输出尽量与真实行为一致。

已经从主接口中移除的典型参数包括：

- `length-hard-threshold`
- `max-processed-length`
- `embed-mid-gpu-threshold`
- `backpressure-low`

同时删除了相关死路径，例如：

- query 压缩比例 `r`
- 旧版 preprocess/compress 函数

### 2.2 在线 EMA 调度器

scheduler 已从“多数逻辑静态、少量逻辑在线”的结构，升级成“在线 feedback 驱动的 dispatch 系统”。

当前 `async_bucket` 的基本行为是：

1. 预计算所有 query 的 token 长度
2. 按 bucket 将 query 放入待处理池
3. 每次 dispatch 时在线决定：
   - bucket
   - batch size
   - `xE/xR`
   - 本批 query 组成
4. 每个 batch 完成后回写运行时 EMA

当前维护的主要统计量包括：

- embedding latency EMA
- retrieval latency EMA
- generation latency EMA
- transfer cost EMA
- batch-size residual EMA

### 2.3 阶段接口重构

embedding 与 retrieval 之间的数据接口已经重写，不再强迫所有 embedding 结果统一落回 CPU。

当前 payload 同时支持：

- `embeddings_cpu`
- `embeddings_gpu`

因此：

- `xE=1, xR=1` 时，可尽量保持 GPU resident
- 避免默认走 `GPU -> CPU -> GPU`

### 2.4 观测与可解释性

`summary_*.json` 已不再只有聚合指标，而是加入了细粒度轨迹信息。

新增输出包括：

- `dispatch_trace`
- `feedback_trace`
- `chunk_trace`
- `timing_breakdown`

这些字段可以用来回答：

- 每次 dispatch 到底选了什么
- chunk 是否真的发生
- 时间主要花在哪一段
- scheduler 自身的开销有多大

### 2.5 Ablation runner

单独新增了 `run_ablation.py`，把多因素消融从 `run_comparison.py` 中分离出来。

当前可控的因素包括：

- 是否使用 bucketing
- 是否开启 online batch sizing
- 是否开启 online action selection
- 是否开启 chunking

原有的 `run_comparison.py` 继续负责标准三模式对比：

- `serial`
- `async_plain`
- `async_bucket`

---

## 3. 关键可靠性修正

### 3.1 早期 ablation 基线错误

第一版 `run_ablation.py` 存在一个关键错误：

- 所有 ablation variant 都被强制用 `--pipeline-mode async_bucket` 运行

这导致：

- `plain_b16`
- `plain_b64`

并不是真正的 plain baseline，而只是“关掉部分功能的 bucket 变体”。

影响：

- 早期关于“plain 大 batch vs bucket”的结论不可靠

修复方式：

- `plain_b16` 和 `plain_b64` 改为使用 `--pipeline-mode async_plain`
- bucket 组继续使用 `--pipeline-mode async_bucket`

这个修正直接改变了后续 ablation 结果的解释。

### 3.2 独立 Agent 验证尝试失败

曾尝试启动独立 agent 对实验代码可靠性做交叉审查，但子 agent 因上游服务错误未返回有效结果。

因此，本轮可靠性检查主要通过以下方式完成：

- 直接检查代码路径
- 检查 summary 各字段之间是否互相一致
- 对比 comparison / ablation 结果是否自洽

---

## 4. Comparison 结果：主要观察

本阶段主要分析了两组 comparison：

- `comparison_large`：普通问题集
- `comparison_long`：超长问题集

### 4.1 普通问题集

主要现象：

- `async_bucket` 明显优于 `serial` 和 `async_plain`

但后续 ablation 说明：

- 这个收益很大一部分并不是“scheduler 本身足够聪明”
- 而是与 batch size 和 generation 吞吐密切相关

### 4.2 超长问题集

主要现象：

- `async_bucket` 与 `async_plain` 大致打平
- 它没有在 long workload 上打开新的收益空间

这说明：

- 当前 bucket / action 逻辑在长问题集上没有形成新的性能优势

---

## 5. Ablation 结果：当前解释

本阶段重点看了两组消融：

- `ablation_large`
- `ablation_long`

### 5.1 `ablation_large` 的强结论

修正 plain baseline 后，普通问题集上的结果变得非常清晰：

1. `plain_b64` 是当前最强方案
2. bucket 系列没有打赢 `plain_b64`
3. online batch sizing 没有展示明显收益
4. online action selection 没有展示明显收益
5. scheduler 开销很小，不是瓶颈

解释：

- 当前最大收益来自更大的 generation batch
- 现阶段 scheduler 的复杂化还没有超过“简单粗暴的大 batch”策略

### 5.2 `ablation_long` 的强结论

超长问题集上的消融结论更直接：

1. `plain_b64` 依然明显优于 bucket 版本
2. bucket 各变体基本都和 `plain_b16` 在同一档，而不是靠近 `plain_b64`
3. online batch sizing 和 online action selection 依然没有明显收益
4. scheduler 开销仍然可以忽略

解释：

- 即使在长问题集上，当前系统的主要性能杠杆依旧是 generation 大 batch
- 当前 bucket 调度并没有把长问题 workload 的优势挖出来

---

## 6. Chunking：当前判断

这里必须严格区分“理论收益”和“实验已证明收益”。

### 6.1 理论上，chunking 解决什么问题

chunking 的理论收益，是相对于 **truncation** 而言，不是相对于“全量 embedding”。

不开 chunk 时：

- 超长 query 会用 `max_length + truncation=True` 路径直接截断

开 chunk 时：

- query 会被切成多个 chunk
- 每个 chunk 分别 embedding
- 再聚合成一个表示

因此，chunk 的理论优势是：

- 相比直接截断，保留更多长 query 的后半段信息

### 6.2 当前实验实际说明了什么

当前实验主要是吞吐实验。

从结果看：

- chunk 往往显著提高 embedding 成本
- 但没有换来明确的 wall-clock 吞吐提升

因此从**性能实验**角度，目前结论是：

- chunk 没有证明自己值得作为吞吐优化手段

但这不等于：

- chunk 对检索质量没有帮助

当前实验只说明：

- 它的性能成本是明确可见的
- 其质量收益尚未被本轮实验验证

---

## 7. Scheduler 开销

当前证据显示：

- scheduler dispatch 开销很小
- scheduler feedback 更新开销也很小
- token length 预处理开销也不大

因此当前可以比较有把握地说：

- scheduler 本身不是瓶颈
- 问题不在“调度器算得太慢”
- 问题在“调度器没有换来足够大的额外收益”

---

## 8. 当前可靠性状态

到目前为止，这套代码已经达到了“可以继续做实验，并且有足够多观测字段来防止自我欺骗”的程度。

当前认为较可靠的部分：

- 主执行路径可以跑通
- summary 里的 trace 字段能互相印证
- ablation baseline 已修正
- 关于“大 batch 是第一性能杠杆”的结论，有多组实验支持

当前尚未建立的东西：

- 当前 online scheduler 接近最优
- 当前 action selection 能充分探索动作空间
- chunk 是否对检索质量有帮助

---

## 9. 当前最可信的总判断

目前最强、最稳定的结论是：

> 对当前测试 workload 而言，最主要的吞吐收益来自更大的 generation batch。  
> 现阶段的 bucket 调度、online batch、online action、chunking，都还没有证明自己能超过一个更简单的 `async_plain + large batch` 基线。

这意味着优化重点需要重新排序。

下一阶段最值得优化的目标，不应该是“继续让当前 bucket scheduler 更复杂”，而应该是：

1. 围绕 generation 大 batch 作为主要目标做系统设计
2. 把异构设备调度的优化围绕“保住大 batch generation 收益”来组织
3. 只有在有额外收益时，才保留 bucketing / chunking

---

## 10. 下一步建议

当前推荐的下一步方向：

1. 将 `plain_b64` 视为当前吞吐主基线
2. 将 scheduler 目标改写为“在设备约束下尽可能形成有效的大 generation batch”
3. 如果未来重新研究 action 调度，优先考虑 bandit 风格的轻探索，而不是继续纯贪心 EMA
4. 如果未来重新研究 chunk，优先把它作为质量问题来单独验证，而不是继续放在吞吐主线上

---

## 10.1 新接入但尚未验证的下一版调度路径

在上述结论基础上，代码中已经开始接入一个新的 scheduler 方向，但**尚未完成实验验证**。

新模式名：

- `generation_target_v1`

新增目的：

- 不再以 `short / mid / long` bucket 为主轴
- 先围绕 generation 大 batch 设定目标
- 再决定 device plan
- 最后才做轻量 batch shaping

当前已接入代码的内容包括：

1. 新增 scheduler 模式开关：
   - `scheduler_mode_choice = legacy_bucket | generation_target_v1`
2. 新增可选 shaping 开关：
   - `--enable-batch-shaping`
3. 新增三组 trace：
   - `generation_target_trace`
   - `device_plan_trace`
   - `batch_shaping_trace`

当前状态判断：

- 这条新路径已经写入代码
- 但还没有完成新的 comparison / ablation 验证
- 因此暂时不能把它视为已验证结论

它应被视为：

- **下一阶段待验证实现**

而不是：

- 当前已经成立的实验结论

---

## 11. 相关文件

核心代码：

- `async_rag_pipeline.py`
- `run_comparison.py`
- `run_ablation.py`

当前文档：

- `docs/pipeline_execution_guide.md`
- `docs/session_code_changes.md`
- `docs/session_experiments.md`
- `docs/session_research_progress.md`

本文档讨论的结果目录：

- `comparison_large`
- `comparison_long`
- `ablation_large`
- `ablation_long`
