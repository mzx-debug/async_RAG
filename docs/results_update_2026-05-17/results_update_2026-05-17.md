# Async RAG Current Results Update

Date: 2026-05-17

## Updated artifacts

- `async_rag_results_visualization_v3.pptx`
- `async_bucket_flow_diagram.svg`
- `async_bucket_flow_diagram.png`

## Current conclusions

### 1. Large group

- `plain_b64` is still the strongest practical baseline.
- Full `async_bucket` is only modestly worse than `plain_b64`.
- The main loss is not from `xE/xR` mis-selection. In practice, both variants mostly use `xE=1, xR=0`.
- The main loss comes from splitting generation into smaller batches after bucketing, which reduces LLM batching efficiency.

### 2. Long group

- Full `async_bucket` is much worse than `plain_b64`.
- The dominant cause is the conservative long-bucket batch policy, which effectively fixes long-query generation around small batches such as `16`.
- This turns what could be `64 + 36` generation into `16,16,16,16,16,16,4`, sharply reducing generation throughput.

### 3. Async bucket interpretation

- The original motivation of using different batch sizes for `short / mid / long` is reasonable for asynchronous overlap.
- However, in the current hardware regime, the system is mostly generation-dominant.
- Because embedding is almost always placed on GPU and becomes cheap, the benefit of stage-time alignment is smaller than the loss from shrinking generation batches.

### 4. Cost model limitation

- The scheduler currently relies on hand-crafted cold-start cost formulas plus EMA updates.
- In high-resource settings, the optimizer has very limited room because the practical optimum is often `large batch + xE=1`.
- On constrained devices, the optimization space should become much more meaningful.

## Suggested next directions

1. In generation-dominant settings, prioritize large generation batches first, then do intra-batch shaping.
2. Move evaluation toward resource-constrained devices where `batch size`, `xE`, and `xR` become real scheduling variables.
3. Focus long-query handling on `CPU embedding` scenarios instead of treating chunking as a default global optimization.
4. Under memory constraints, jointly optimize remaining GPU memory, generation batch size, and `xE/xR` choices.

## Notes

- The updated PPTX uses improved ablation visualizations with horizontal bar layouts to avoid label overlap.
- The flow diagram is available in both SVG and high-resolution PNG form for slide use.
