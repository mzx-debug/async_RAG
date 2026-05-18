| mode | execution | num_queries | wall_time_ms | wall_qps | total_ms(sum stages) | avg_emb_ms | avg_ret_ms | avg_gen_ms |
|---|---|---|---|---|---|---|---|---|
| serial | serial_pipeline | 512 | 93437.07 | 5.4796 | 93413.26 | 2.0334 | 18.1920 | 162.2224 |
| async_plain | async_threaded_pipeline_plain | 512 | 84188.21 | 6.0816 | 99443.16 | 13.1983 | 17.4175 | 163.6091 |
| async_bucket | async_threaded_pipeline_bucket | 512 | 37927.10 | 13.4996 | 44966.22 | 1.3315 | 14.5035 | 71.9897 |