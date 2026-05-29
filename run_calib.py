#!/usr/bin/env python3
"""Standalone calibration: run pipeline at multiple batch sizes and fit P/D."""
import sys, os, json, logging, time
sys.path.insert(0, '/home/cloudteam/rag_mzx')
os.environ.update({
    'HF_ENDPOINT': 'https://hf-mirror.com',
    'LD_PRELOAD': '',
    'MKL_THREADING_LAYER': 'GNU',
    'OPENBLAS_NUM_THREADS': '4',
    'OMP_NUM_THREADS': '4',
    'LD_LIBRARY_PATH': '/usr/local/cuda-12.8/lib64',
})
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import importlib.util
spec = importlib.util.spec_from_file_location(
    'pipeline_mod', '/home/cloudteam/rag_mzx/async_rag_pipeline.py'
)
pipeline_mod = importlib.util.module_from_spec(spec)
sys.modules['pipeline_mod'] = pipeline_mod
spec.loader.exec_module(pipeline_mod)
StandaloneRAGPipeline = pipeline_mod.StandaloneRAGPipeline


class Args:
    index_path = '/home/cloudteam/rag_mzx/indexes/beir_nfcorpus/faiss_hnsw.index'
    corpus_path = '/home/cloudteam/rag_mzx/data/beir_nfcorpus/corpus.jsonl'
    generator_model = 'Qwen/Qwen2.5-1.5B-Instruct'
    b = 4; xE = 0; xR = 0; nprobe = 128; index_type = 'hnsw'
    topk = 1; embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_max_length = 384; pooling_method = 'mean'
    embedding_use_fp16 = False; max_output_len = 32; max_model_len = None
    temperature = 0.0; top_p = 1.0; top_k = None
    prompt_template = 'Question: {query}\nContext: {context}\nAnswer:'
    tensor_parallel_size = 1; gpu_memory_utilization = 0.8
    fixed_action = True; vllm_enforce_eager = False; gpu_id = None
    queries_file = '/home/cloudteam/rag_mzx/data/beir_nfcorpus/queries_beir.jsonl'
    query_field = 'question'; dataset_name = 'beir_nfcorpus'; dataset_split = 'test'
    sample_queries = 256; seed = 2026
    log_interval = 9999; show_samples = 0; output_json = None
    pipeline_mode = 'async_v2'; embed_long_gpu_threshold = 128
    retrieve_gpu_batch_threshold = 64; enable_memory_aware_scheduling = True
    gpu_mem_low_threshold_gb = 4.0; gpu_mem_medium_threshold_gb = 10.0
    gpu_mem_high_batch_penalty = 50.0; faiss_index_gb = 2.0
    generator_model_layers = None; generator_model_hidden = None
    backpressure_high = 8; scheduler_ema_alpha = 0.25
    initial_batch_size = 32; ablate_online_batch = False
    ablate_online_action = False; ablate_chunking = False
    ema_params_path = None; save_ema_params = False; run_calibration = False


if __name__ == '__main__':
    batch_sizes = [4, 64, 256]
    # First run gets 0.8 util; subsequent runs get 0.5 to share GPU memory
    gpu_utils = [0.8, 0.5, 0.5]
    results = {}

    for i, b in enumerate(batch_sizes):
        out_path = f'/home/cloudteam/rag_mzx/output/calib_v2_{b}.json'
        args = Args()
        args.b = b
        args.output_json = out_path
        args.gpu_memory_utilization = gpu_utils[i]

        print(f"\n=== B={b} (gpu_util={gpu_utils[i]}) ===")
        pipe = StandaloneRAGPipeline(args)
        summary = pipe.run()

        results[b] = {
            'batch_size': b,
            'avg_gen_ms': summary['avg_generation_ms'],
            'avg_emb_ms': summary['avg_embedding_ms'],
            'avg_ret_ms': summary['avg_retrieval_ms'],
            'total_ms': summary['total_ms'],
            'throughput_qps': summary['throughput_qps'],
            'gen_base': dict(pipe.scheduler._gen_base_overhead_ema),
            'gen_per_q': dict(pipe.scheduler._gen_per_query_ema),
            'contention': dict(pipe.scheduler._contention_ema),
            'startup_k': dict(pipe.scheduler._startup_k_ema),
            'dispatch_trace': summary.get('dispatch_trace', []),
            'feedback_trace': summary.get('feedback_trace', []),
        }

        print(f"  gen={summary['avg_generation_ms']:.1f}ms/q "
              f"emb={summary['avg_embedding_ms']:.1f}ms "
              f"ret={summary['avg_retrieval_ms']:.2f}ms "
              f"qps={summary['throughput_qps']:.1f}")

        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)

        time.sleep(3)

    # Summary table
    print("\n" + "=" * 60)
    print("  CALIBRATION RESULTS")
    print("=" * 60)
    print("  B     | gen_ms | emb_ms | ret_ms")
    print("  " + "-" * 30)
    for b, r in results.items():
        print(f"  {b:5d} | {r['avg_gen_ms']:6.1f} | {r['avg_emb_ms']:5.1f} | {r['avg_ret_ms']:5.2f}")

    # Two-point fit
    import numpy as np
    Bs = np.array([r['batch_size'] for r in results.values()], dtype=float)
    gens = np.array([r['avg_gen_ms'] for r in results.values()], dtype=float)
    gen_qs = gens / Bs

    P = (gen_qs[0] - gen_qs[1]) / (1.0/Bs[1] - 1.0/Bs[0])
    D = gen_qs[0] - P / Bs[0]
    P = max(0.0, P)
    D = max(0.0, D)

    print(f"\n  Fitted model: gen_q = {P:.0f}/B + {D:.1f}")
    print(f"\n  Verification:")
    print(f"  B     | pred_ms | obs_ms  | err%")
    print("  " + "-" * 35)
    for b, r in results.items():
        pred = P/b + D
        obs = r['avg_gen_ms']
        err = (obs - pred) / obs * 100
        print(f"  {b:5d} | {pred:7.1f} | {obs:7.1f} | {err:+.0f}%")

    # Contention at (0,0) from runtime EMA
    cont_vals = list(results[batch_sizes[-1]]['contention'].values())
    cont_00 = cont_vals[0] if cont_vals else 1.0
    print(f"\n  Contention (0,0) after B={batch_sizes[-1]}: {cont_00:.4f}")

    # Save summary
    with open('/home/cloudteam/rag_mzx/output/calibration_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: output/calibration_summary.json")
