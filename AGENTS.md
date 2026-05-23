# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-first Python experiment for comparing RAG execution modes. Keep core pipeline logic in `async_rag_pipeline.py`, index construction in `build_index.py`, and experiment orchestration in `run_comparison.py`. Input data lives in `data/`. Generated reports and summaries currently live in `comparison/` and `comparison_large/`. Working notes and run instructions belong in `docs/`.

When adding new code, prefer small helper functions over new top-level scripts unless the workflow is truly standalone.

## Build, Test, and Development Commands
Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key workflows:

```bash
python ./build_index.py --corpus-path ./data/corpus.jsonl --output-dir ./indexes/flat --device cuda
python ./async_rag_pipeline.py --pipeline-mode async_bucket --index-path ./indexes/flat/faiss.index --corpus-path ./data/corpus.jsonl --generator-model meta-llama/Llama-3.1-8B-Instruct --queries-file ./data/queries_generated.jsonl
python ./run_comparison.py --workdir . --index-path ./indexes/flat/faiss.index --corpus-path ./data/corpus.jsonl --generator-model meta-llama/Llama-3.1-8B-Instruct --queries-file ./data/queries_generated.jsonl --output-dir ./comparison
```

Use `docs/pipeline_execution_guide.md` for fuller parameter examples.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and clear docstrings for public helpers. Follow existing Python naming: `snake_case` for functions and variables, `PascalCase` for classes, and descriptive CLI flags such as `--queries-file` and `--pipeline-mode`.

Keep scripts importable and avoid burying logic inside `main()`. Prefer standard library utilities and keep new dependencies justified in `requirements.txt`.

## Testing Guidelines
There is no dedicated `tests/` directory yet. Validate changes by running the smallest relevant script path first, then a full comparison if behavior affects scheduling or metrics. Check generated JSON summaries and Markdown tables for schema stability and sensible metric deltas.

If you add reusable logic, introduce `pytest` tests under `tests/` with names like `test_scheduler.py`.

## Commit & Pull Request Guidelines
Git history is not available in this workspace snapshot, so use short imperative commit subjects, for example: `pipeline: tighten async bucket scheduling`. Keep one logical change per commit.

Pull requests should include:
- A short summary of the experiment or code change
- Exact commands used for validation
- Notes on data, model, or GPU assumptions
- Output snippets or tables when metrics change

## Configuration & Data Tips
Do not commit large generated indexes, model weights, or raw experiment dumps unless they are required artifacts. Keep dataset paths configurable by CLI flags and document non-default hardware assumptions in `docs/`.
