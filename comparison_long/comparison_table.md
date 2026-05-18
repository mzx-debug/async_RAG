| mode | execution | num_queries | wall_time_ms | wall_qps | total_ms(sum stages) | avg_emb_ms | avg_ret_ms | avg_gen_ms |
|---|---|---|---|---|---|---|---|---|
| serial | serial_pipeline | 100 | 22041.12 | 4.5370 | 22035.07 | 4.6731 | 19.9444 | 195.7332 |
| async_plain | async_threaded_pipeline_plain | 100 | 20035.17 | 4.9912 | 22064.08 | 5.7542 | 18.1803 | 196.7063 |
| async_bucket | async_threaded_pipeline_bucket | 100 | 20098.10 | 4.9756 | 22448.92 | 8.6963 | 18.9439 | 196.8490 |