# 待办事项

最后更新：2026-05-23

---

## 立即

- [ ] 在资源受限场景下运行三模式对比（`gpu_memory_utilization=0.3`），验证显存感知调度的效果
  - 包含：显存感知 action 选择、显存感知桶优先级、lookahead dispatch
  - 预期：`async_bucket` 在低显存场景下比 `plain_b64` 表现更好

---

## 短期（1 周内）

### 资源受限实验

- [ ] ResourceTracker 行为验证
  - 观察 `resource_tracker.vllm_reserved_gb` 是否合理（应为 gpu_total * gpu_memory_utilization）
  - 观察 `_detect_gpu_free_memory_gb()` 在 pipeline 运行期间的变化

- [ ] 显存感知 action 选择验证
  - 低显存（<4 GiB）时 `xE=0, xR=0` 是否被优先选择
  - 中显存（4-10 GiB）时 `xE=1, xR=0` 是否最优
  - 通过 `dispatch_trace` 确认 action_counts 分布符合预期

- [ ] 显存压力感知的桶优先级验证
  - 高显存压力：long 查询是否被优先调度
  - 低显存压力：short 查询是否保持高优先级
  - 通过 `dispatch_trace` 中的 bucket 分布确认

- [ ] Lookahead dispatch 效果验证
  - `--enable-lookahead-dispatch` 开关对比：启用 vs 禁用
  - 观察 `max_q_er` 是否在有 lookahead 时更高（pipeline 填得更满）
  - 对比 wall_time 差异

- [ ] 联合决策效果验证
  - `plain_b64` vs `async_bucket + 显存感知` 在低显存场景下的 QPS 差距
  - 确认显存感知版本在资源受限时不再输 `plain_b64`

### 参数调优

- [ ] 调整 `--gpu-mem-low-threshold-gb` 和 `--gpu-mem-medium-threshold-gb`
  - 根据实际 GPU 规格（总显存）重新设定阈值
  - 建议：总显存 24 GiB → high<6, medium<12；总显存 40 GiB → high<8, medium<16

- [ ] 调整 `--gpu-mem-high-batch-penalty`
  - 50.0 分可能过重，导致即使在中等显存压力下也无法使用 xR=1
  - 建议先用 `--gpu-mem-high-batch-penalty 30.0` 跑一次看效果

---

## 中期（2-4 周）

- [ ] nprobe 对比实验（资源受限场景）
  - nprobe=32/128/512，显存受限下 async 收益曲线
  - 目标：找到 retrieval 占比对显存受限下调度收益的影响

- [ ] vLLM gpu_memory_utilization 梯度实验
  - gpu_util=0.3/0.4/0.5/0.8 下对比 async_bucket vs plain_b64
  - 目标：找到显存利用率阈值，超过后显存感知调度收益归零

- [ ] CPU embed 在显存受限场景的效果
  - `--xE 0 --xR 0` 在低显存场景下是否反而比 `--xE 1 --xR 0` 更快（避免 GPU embedding 争抢显存）

- [ ] FAISS index 分片策略
  - 当 FAISS index 无法全驻留 GPU 时，部分驻留策略
  - 目标：在显存受限场景下找到 retrieval 性能与显存占用的最优平衡

- [ ] 动态显存感知 batch shaping
  - 根据实时显存状态，在运行时调整 batch size 上限
  - 替代当前的静态 `_bucket_batch_size` 逻辑

---

## 长期（研究方向）

- [ ] 资源受限场景下的最优调度策略形式化
  - 目标：找到"在 GPU 显存约束下，最大化 generation throughput"的理论最优解
  - 与资源充裕场景的策略对比，形成完整的调度策略图谱

- [ ] 写论文/报告
  - 主题：资源受限场景下的异步 RAG 流水线调度
  - 核心贡献：
    1. 显存感知调度的必要性证明（vs 充裕场景对比）
    2. lookahead dispatch 的理论分析
    3. 显存压力驱动的桶优先级策略

---

## 已完成

- [x] 修复 stage_cache 预填充 bug（embedding 重复加载模型）
- [x] 添加 warmup 机制（三模式公平对比）
- [x] 实现长 query 切分 embedding（64 token chunk + 平均向量）
- [x] 切片嵌入改为仅 async_bucket 模式启用（对比公平性）
- [x] 支持 Arrow 格式语料库（load_from_disk）
- [x] 生成 512 条问题集（queries_generated.jsonl）
- [x] 生成 100 条超长问题集（queries_long.jsonl，128-226 token）
- [x] 完成小库三模式对比实验
- [x] 下载大语料库（msmarco-passage-corpus，880 万条）
- [x] 下载大索引（ivf.index，34GB）
- [x] 完成大库三模式对比实验（含修复后重跑）
- [x] 修复 Bug 1：调度器 action 越界 → 改为运行时过滤（开放 action 空间）
- [x] 修复 Bug 2：short 桶被强制 CPU embedding（删除过滤条件）
- [x] 修复 Bug 3：FAISS 默认单线程（添加 omp_set_num_threads + 边缘策略）
- [x] 修复 Bug 4：大 batch retrieval 无中间日志（分片 + 日志）
- [x] 修复 Bug 5：GPU 显存检查用总显存而非可用显存
- [x] 新增边缘设备 FAISS 线程策略（`--faiss-omp-threads` + 自动保守档）
- [x] 添加 tqdm 进度条（serial + async 模式）
- [x] 改进日志输出（每批打印耗时、进度、ETA）
- [x] run_comparison.py 透传子进程输出
- [x] 实现动态 batch_size（hill-climbing 延迟反馈）
- [x] 开放 action 空间（运行时过滤替代 CLI 限制）
- [x] Per-Batch Action 记录（BatchStats 新增 xE/xR 字段）
- [x] 实现 ResourceTracker 类（GPU 显存实时监控 + 各阶段预估）
- [x] 改造 `_action_feasible()`（显存感知 action 过滤与打分）
- [x] 改造 `_bucket_priority()`（显存压力驱动的桶优先级）
- [x] 实现 lookahead dispatch（显存压力感知的提前 dispatch）
- [x] 新增显存感知 CLI 参数（memory-aware、lookahead、thresholds）
- [x] 整合 ResourceTracker 到 StandaloneRAGPipeline
