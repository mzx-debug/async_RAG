#!/usr/bin/env python3
"""
Verify action-selection model by comparing (0,0) vs (0,1) vs (1,0) vs (1,1)
at the same batch size and gpu_util.
Forces fixed actions via --fixed-action flag, measures actual cost.
"""
import argparse, json, os, random, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PYTHON = "/home/cloudteam/Software/conda/envs/p702/bin/python"
OUT_DIR = ROOT / "output" / "action_verification"
TIMEOUT = 300
QUERIES = 200

CORPUS_CONFIG = {
    "beir_nfcorpus": {
        "corpus_path": str(ROOT / "data" / "beir_nfcorpus" / "corpus.jsonl"),
        "queries_path": str(ROOT / "data" / "beir_nfcorpus" / "queries.jsonl"),
        "index_path": str(ROOT / "indexes" / "beir_nfcorpus" / "faiss_flat.index"),
    },
}


def kill_gpu_residuals():
    """Kill leftover VLLM/Python processes consuming GPU memory."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.strip().split("\n"):
            if not line:
                continue
            pid_str = line.split(",")[0].strip()
            try:
                pid = int(pid_str)
                if pid != os.getpid():
                    subprocess.run(["kill", "-9", str(pid)], capture_output=True, timeout=5)
            except (ValueError, IndexError):
                pass
        time.sleep(2)
    except Exception:
        pass


def load_queries_short(path: str, n: int) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("token_length", 0) <= 8:
                records.append(rec)
    random.seed(42)
    random.shuffle(records)
    return records[:n]


def build_queries() -> str:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    queries = load_queries_short(cfg["queries_path"], QUERIES * 2)
    out = OUT_DIR / "queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return str(out)


@dataclass
class RunResult:
    action: tuple  # (xE, xR)
    gpu_util: float
    batch_size: int
    total_ms: float
    wall_ms: float
    throughput: float
    gen_pt: float
    avg_gen_ms_q: float
    avg_emb_ms_q: float
    avg_ret_ms_q: float


def run_pipeline(xe: int, xr: int, gpu_util: float, batch_size: int, queries_file: str) -> RunResult:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    out_json = OUT_DIR / f"xe{xe}_xr{xr}_gpu{gpu_util}_b{batch_size}.json"
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    # Clean GPU residuals before each run
    kill_gpu_residuals()

    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", "async_plain",
        "--xE", str(xe), "--xR", str(xr),
        "--b", str(batch_size),
        "--fixed-action",
        "--index-path", cfg["index_path"],
        "--corpus-path", cfg["corpus_path"],
        "--queries-file", queries_file,
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_json),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_util),
        "--gpu-id", str(gpu_id),
        "--sample-queries", str(QUERIES),
    ]
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["PYTHONUNBUFFERED"] = "1"

    label = f"(xE={xe},xR={xr}) gpu={gpu_util} B={batch_size}"
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
        elapsed = time.time() - t0
        if result.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                d = json.load(f)
            wall_ms = d.get("wall_time_ms", 0)
            throughput = d.get("wall_throughput_qps", 0)
            gen_pt = d.get("generation_ms_per_token", 0)
            avg_gen = d.get("avg_generation_ms", 0)
            avg_emb = d.get("avg_embedding_ms", 0)
            avg_ret = d.get("avg_retrieval_ms", 0)
            total_ms = d.get("total_ms", 0)
            print(f"OK ({elapsed:.0f}s)")
            return RunResult(
                action=(xe, xr), gpu_util=gpu_util, batch_size=batch_size,
                total_ms=total_ms, wall_ms=wall_ms, throughput=throughput,
                gen_pt=gen_pt, avg_gen_ms_q=avg_gen, avg_emb_ms_q=avg_emb, avg_ret_ms_q=avg_ret,
            )
        print(f"FAIL ({elapsed:.0f}s)")
        if result.stdout:
            print(f"  stdout: {result.stdout[-300:]}")
        if result.stderr:
            print(f"  stderr: {result.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    return RunResult(
        action=(xe, xr), gpu_util=gpu_util, batch_size=batch_size,
        total_ms=0, wall_ms=0, throughput=0, gen_pt=0,
        avg_gen_ms_q=0, avg_emb_ms_q=0, avg_ret_ms_q=0,
    )


# ── Original cost model (unchanged from codebase) ─────────────────────────────────

# Model prediction — Qwen2.5-1.5B-Instruct实测参数 (from measure_gpu_costs.py)
L = 3.23
avg_out = 120.0
gen_pt = 0.2135      # ms/token，实测
gen_base = 1109.0    # ms prefill overhead，实测
queue = 0.23         # ms/q 调度开销
queue = 0.23
e0, oh_e0 = 0.36, 2.71
e1, oh_e1 = 0.00, 4.87
r0, oh_r0 = 0.20, 1.36
r1, oh_r1 = 0.00, 1.50
K01 = 0.55


def predict(xe: int, xr: int, B: int) -> float:
    gen_q = gen_pt * avg_out + gen_base / B
    cpu_q = (e0 * L + oh_e0 / B if xe == 0 else 0) + (r0 + oh_r0 / B if xr == 0 else 0)
    gpu_q = gen_q + (e1 * L + oh_e1 / B if xe == 1 else 0) + (r1 + oh_r1 / B if xr == 1 else 0)
    xfer = K01 * L if xe != xr else 0
    return max(cpu_q, gpu_q) + queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-utils", default="0.5,0.8")
    parser.add_argument("--batch-sizes", default="8,32")
    args = parser.parse_args()
    gpu_utils = [float(x) for x in args.gpu_utils.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    actions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    queries_file = build_queries()
    results = {}

    for gpu_util in gpu_utils:
        results[gpu_util] = {}
        for B in batch_sizes:
            results[gpu_util][B] = {}
            for xe, xr in actions:
                r = run_pipeline(xe, xr, gpu_util, B, queries_file)
                results[gpu_util][B][(xe, xr)] = r

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  ACTION VERIFICATION: Predicted vs Actual Cost")
    print(f"{'='*90}")

    for gpu_util in sorted(results.keys()):
        print(f"\ngpu_util={gpu_util}:")
        for B in batch_sizes:
            print(f"\n  B={B}:")
            print(f"  {'Action':<10} | {'pred_ms':>10} | {'actual_ms/q':>12} | {'err%':>8} | "
                  f"{'emb':>8} | {'ret':>8} | {'gen':>8} | {'gen_pt':>8}")
            print(f"  {'-'*80}")

            actuals = []
            preds = []
            for xe, xr in actions:
                r = results[gpu_util][B][(xe, xr)]
                if r.wall_ms == 0:
                    print(f"  ({xe},{xr})      | {'FAIL':>10}")
                    continue
                pred = predict(xe, xr, B)
                actual_q = r.wall_ms / QUERIES
                err = (pred - actual_q) / actual_q * 100 if actual_q > 0 else float("inf")
                print(f"  ({xe},{xr})      | {pred:>10.1f} | {actual_q:>12.1f} | {err:>+7.1f}% | "
                      f"{r.avg_emb_ms_q:>8.2f} | {r.avg_ret_ms_q:>8.2f} | "
                      f"{r.avg_gen_ms_q:>8.1f} | {r.gen_pt:>8.4f}")
                actuals.append((xe, xr, actual_q))
                preds.append((xe, xr, pred))

            # Ranking correlation
            if actuals:
                actual_rank = sorted(actuals, key=lambda x: x[2])
                pred_rank = sorted(preds, key=lambda x: x[2])
                correct = sum(1 for (a, _, av), (p, _, pv) in zip(actual_rank, pred_rank) if a == p)
                kendall_tau = (correct - (len(actions) - correct)) / len(actions)
                print(f"  Ranking: correct={correct}/{len(actuals)} (Kendall tau={kendall_tau:+.2f})")

                # Best action
                best_actual = min(actuals, key=lambda x: x[2])
                best_pred = min(preds, key=lambda x: x[2])
                print(f"  Best: actual=({best_actual[0]},{best_actual[1]}) {best_actual[2]:.1f}ms/q  "
                      f"pred=({best_pred[0]},{best_pred[1]}) {best_pred[2]:.1f}ms/q  "
                      f"{'✓' if best_actual[0]==best_pred[0] and best_actual[1]==best_pred[1] else '✗'}")

    # Save — serialize tuple keys as strings
    serializable = {}
    for gpu_util, by_B in results.items():
        serializable[str(gpu_util)] = {}
        for B, by_act in by_B.items():
            serializable[str(gpu_util)][str(B)] = {}
            for (xe, xr), r in by_act.items():
                serializable[str(gpu_util)][str(B)][f"({xe},{xr})"] = {
                    "total_ms": r.total_ms, "wall_ms": r.wall_ms,
                    "throughput": r.throughput, "gen_pt": r.gen_pt,
                    "avg_gen_ms_q": r.avg_gen_ms_q, "avg_emb_ms_q": r.avg_emb_ms_q,
                    "avg_ret_ms_q": r.avg_ret_ms_q,
                    "pred_ms": predict(xe, xr, B),
                }

    out = OUT_DIR / "action_results.json"
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
