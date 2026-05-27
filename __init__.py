"""
Async RAG Pipeline V1 — Resource-Constrained Edition

A standalone RAG pipeline that compares serial, async_plain, and async_v2
scheduling strategies on resource-constrained hardware (4–16 GB VRAM).

Example usage:
    from async_rag_pipeline import StandaloneRAGPipeline, build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--pipeline-mode", "async_v2",
        "--index-path", "./indexes/faiss.index",
        "--corpus-path", "./data/corpus.jsonl",
        "--generator-model", "Qwen/Qwen2.5-3B-Instruct",
        "--b", "32", "--xE", "1", "--xR", "0",
    ])
    pipeline = StandaloneRAGPipeline(args)
    summary = pipeline.run()
"""

__version__ = "1.0.0"
__author__ = "Async RAG Team"

from async_rag_pipeline import (
    build_parser,
    GreedyScheduler,
    ResourceTracker,
    QueryEmbeddingStage,
    RetrievalStage,
    GenerationStage,
    StandaloneRAGPipeline,
)

__all__ = [
    "build_parser",
    "GreedyScheduler",
    "ResourceTracker",
    "QueryEmbeddingStage",
    "RetrievalStage",
    "GenerationStage",
    "StandaloneRAGPipeline",
]
