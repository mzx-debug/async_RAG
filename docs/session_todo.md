# 待办事项

最后更新：2026-05-12

---

## 立即

- [ ] 重跑三模式对比（大库），验证所有改动的综合效果
  - 包含：action 空间开放、动态 batch_size、切片嵌入仅限 bucket、per-batch action 记录
  - 预期：async_bucket wall_time < 38s，per_batch 中可见 xE/xR/r 字段

---

## 短期（1 周内）

- [ ] 长 query 切分效果验证
  - 用 `data/queries_long.jsonl`（100 条，128-226 token）
  - 对比 async_bucket（有切片）vs serial（无切片）的检索质量
  - 确认切片嵌入不引入 OOM 或性能退化

- [ ] 在线延迟跟踪替代静态评分
  - 用 EMA 跟踪每个 (bucket, action) 组合的实际 per_query_latency
  - 冷启动用现有静态公式作为 prior
  - 解决 comparison 实验中 nprobe=32 时错误选择 xE0_xR1 的问题

- [ ] nprobe 对比实验
  - nprobe = 32 / 128 / 512，serial 模式，大库
  - 目标：找到 retrieval 占比随 nprobe 的变化曲线

- [ ] 动态 batch_size 收敛验证
  - 用 2000+ 条 query 跑 async_bucket
  - 观察 _batch_state 的 size 变化轨迹
  - 确认 3-5 个 batch 后收敛

- [ ] FAISS 线程数扫描（边缘设备）
  - 在大库场景固定其他参数，测试 `--faiss-omp-threads = 1 / 2 / 4 / 8`
  - 记录 wall_time、QPS、CPU 占用
  - 目标：确定"边缘设备默认推荐线程数"

---

## 中期（2-4 周）

- [ ] 绘制 retrieval 占比 vs async 加速比曲线
  - X 轴：retrieval 占比（通过 nprobe 控制）
  - Y 轴：async_plain 和 async_bucket 相对 serial 的加速比
  - 目标：量化 async 流水线的适用边界

- [ ] 探索 generation batch 动态调整
  - 当前：batch size 以 query 数为单位
  - 目标：以 token 数为单位打包 batch，让每个 generation batch 的 token 总量接近 vLLM 最优吞吐点

- [ ] 生成更长的问题集（>256 token）
  - 当前 queries_long.jsonl 最长只有 226 token
  - 需要生成 300-500 token 的问题集，充分测试多 chunk 切分

---

## 长期（研究方向）

- [ ] 量化 async 流水线的适用边界
  - 结论形式："当 retrieval 占比 > X% 时，async 才有 Y% 的收益"
  - 对应论文场景（TELERAG 1.53×，PipeRAG 2.6×）的条件

- [ ] 写论文/报告
  - 主题：async RAG pipeline 的适用边界与优化策略
  - 核心贡献：量化 retrieval 占比对 async 收益的影响 + 动态调度策略

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
- [x] Per-Batch Action 记录（BatchStats 新增 xE/xR/r 字段）
