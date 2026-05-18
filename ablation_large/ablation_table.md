| name | mode | wall_time_ms | wall_qps | avg_emb_ms | avg_ret_ms | avg_gen_ms | dispatch_ms | feedback_ms |
|---|---|---|---|---|---|---|---|---|
| plain_b16 | plain_fixed_batch | 85339.09 | 5.9996 | 13.4572 | 18.6825 | 165.2450 | 0.0000 | 1.4793 |
| plain_b64 | plain_fixed_batch | 33945.15 | 15.0832 | 2.0293 | 14.2803 | 64.3552 | 0.0000 | 0.5505 |
| bucket_fixed_batch_fixed_action | online_dispatch_ema_v1 | 37968.75 | 13.4848 | 1.5102 | 14.7004 | 72.2224 | 1.1376 | 0.4774 |
| bucket_online_batch_fixed_action | online_dispatch_ema_v1 | 37982.35 | 13.4799 | 1.5310 | 14.4523 | 72.2565 | 2.6960 | 0.6514 |
| bucket_online_batch_online_action | online_dispatch_ema_v1 | 37997.46 | 13.4746 | 1.5301 | 14.8136 | 72.1607 | 15.0247 | 0.4181 |
| bucket_online_batch_online_action_no_chunk | online_dispatch_ema_v1 | 38765.25 | 13.2077 | 1.0764 | 15.4348 | 73.7561 | 1.7598 | 0.9288 |