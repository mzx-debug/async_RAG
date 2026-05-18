| name | mode | wall_time_ms | wall_qps | avg_emb_ms | avg_ret_ms | avg_gen_ms | dispatch_ms | feedback_ms |
|---|---|---|---|---|---|---|---|---|
| plain_b16 | plain_fixed_batch | 20117.58 | 4.9708 | 4.3214 | 19.3107 | 196.6335 | 0.0000 | 0.2378 |
| plain_b64 | plain_fixed_batch | 10126.17 | 9.8754 | 4.2850 | 14.4812 | 89.2427 | 0.0000 | 0.0930 |
| bucket_fixed_batch_fixed_action | online_dispatch_ema_v1 | 19981.54 | 5.0046 | 10.5877 | 19.3814 | 195.4069 | 0.6146 | 0.2337 |
| bucket_online_batch_fixed_action | online_dispatch_ema_v1 | 20183.48 | 4.9545 | 10.3217 | 21.7532 | 197.5665 | 0.7770 | 0.2818 |
| bucket_online_batch_online_action | online_dispatch_ema_v1 | 20203.35 | 4.9497 | 9.3882 | 22.7960 | 197.3838 | 0.9016 | 0.2708 |
| bucket_online_batch_online_action_no_chunk | online_dispatch_ema_v1 | 19993.03 | 5.0017 | 3.9263 | 18.3470 | 196.0912 | 0.8891 | 0.3227 |