#!/usr/bin/env python3

import argparse
import heapq
import json
import logging
import math
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_primary_gpu_id(raw_gpu_id: Optional[Any]) -> int:
    if raw_gpu_id is None:
        return 0

    text = str(raw_gpu_id).strip()
    if not text:
        return 0
    first_token = text.split(",")[0].strip()
    if not first_token:
        return 0

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        visible_tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if first_token in visible_tokens:
            return visible_tokens.index(first_token)

        # When only one visible device is exposed, its local ordinal is always 0.
        if len(visible_tokens) == 1:
            return 0

    return int(first_token)


def synchronize_cuda_if_needed(device_index: Optional[int] = None) -> None:
    if not torch.cuda.is_available():
        return
    if device_index is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device=device_index)


def pooling(
    pooler_output: Optional[torch.Tensor],
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_method: str,
) -> torch.Tensor:
    if pooling_method == "mean":
        masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        if pooler_output is None:
            raise ValueError("pooler_output is None, but pooling_method='pooler'.")
        return pooler_output
    raise ValueError(f"Unsupported pooling method: {pooling_method}")


def extract_question(record: Dict[str, Any], preferred_field: str) -> Optional[str]:
    if preferred_field in record:
        value = record.get(preferred_field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for field in ("question", "query", "question_text", "text"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_field in ("text", "question"):
                nested_value = value.get(nested_field)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    return None


def extract_doc_text(doc: Dict[str, Any]) -> str:
    if "contents" in doc and doc["contents"] is not None:
        return str(doc["contents"])
    if "text" in doc and doc["text"] is not None:
        if "title" in doc and doc["title"]:
            return f"{doc['title']}\n{doc['text']}"
        return str(doc["text"])
    if "title" in doc and doc["title"] is not None:
        return str(doc["title"])
    return json.dumps(doc, ensure_ascii=False)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_queries(args: argparse.Namespace) -> List[str]:
    if args.queries_file is not None:
        query_path = Path(args.queries_file).expanduser().resolve()
        if not query_path.is_file():
            raise FileNotFoundError(f"queries_file not found: {query_path}")
        suffix = query_path.suffix.lower()

        if suffix == ".txt":
            with query_path.open("r", encoding="utf-8") as handle:
                queries = [line.strip() for line in handle if line.strip()]
        elif suffix in (".jsonl", ".json"):
            if suffix == ".jsonl":
                records = load_jsonl(query_path)
            else:
                content = json.loads(query_path.read_text(encoding="utf-8"))
                if isinstance(content, list):
                    records = content
                elif isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
                    records = content["data"]
                else:
                    raise ValueError("JSON query file must be a list or a dict with a 'data' list.")

            queries = []
            for record in records:
                if isinstance(record, str):
                    query = record.strip()
                elif isinstance(record, dict):
                    query = extract_question(record, args.query_field)
                else:
                    query = None
                if query:
                    queries.append(query)
        else:
            raise ValueError("queries_file must be .txt, .jsonl, or .json")
    else:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        queries = []
        for row in dataset:
            if not isinstance(row, dict):
                continue
            query = extract_question(row, args.query_field)
            if query:
                queries.append(query)

    if not queries:
        raise ValueError("No queries loaded.")

    if args.sample_queries is not None and args.sample_queries > 0 and args.sample_queries < len(queries):
        rng = np.random.RandomState(args.seed)
        sampled = rng.choice(len(queries), size=args.sample_queries, replace=False)
        sampled = sorted(sampled.tolist())
        queries = [queries[i] for i in sampled]

    return queries


def load_corpus(corpus_path: str) -> Any:
    path = Path(os.path.expandvars(os.path.expanduser(corpus_path)))
    if path.exists():
        if path.suffix.lower() == ".jsonl":
            return load_jsonl(path)
        if path.suffix.lower() == ".json":
            content = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(content, list):
                return content
            if isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
                return content["data"]
            raise ValueError("JSON corpus file must be a list or a dict with a 'data' list.")
        # Arrow dataset directory saved via Dataset.save_to_disk()
        if path.is_dir() and (path / "state.json").exists():
            from datasets import load_from_disk
            return load_from_disk(str(path))
        return load_dataset("json", data_files=str(path), split="train")

    dataset = load_dataset(corpus_path)
    if "train" in dataset:
        return dataset["train"]
    first_split = next(iter(dataset.keys()))
    return dataset[first_split]


class QueryEmbeddingStage:
    def __init__(
        self,
        model_path: str,
        pooling_method: str,
        max_length: int,
        backend: str,
        use_fp16: bool,
        gpu_id: int,
        chunked_embedding: bool = False,
    ) -> None:
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.backend = backend
        self.chunked_embedding = chunked_embedding
        self.cuda_device_index: Optional[int] = None
        self.last_chunk_stats: Optional[Dict[str, Any]] = None

        if backend == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("xE=1 requires CUDA, but CUDA is not available.")
            self.device = torch.device("cuda")
            self.cuda_device_index = gpu_id
        elif backend == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        if self.device.type == "cuda" and use_fp16:
            self.model = self.model.half()
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        queries: Sequence[str],
        output_target: str = "cpu",
    ) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor], float]:
        start = time.perf_counter()
        chunk_stats = {
            "num_queries": len(queries),
            "num_chunked_queries": 0,
            "total_chunks": len(queries),
            "avg_chunks_per_query": 1.0 if queries else 0.0,
            "max_chunks_for_one_query": 1 if queries else 0,
        }

        if self.chunked_embedding:
            # Split each query into token-level chunks to avoid truncation loss.
            # chunk_size=64 keeps each chunk semantically focused for dense retrieval.
            # Queries within 64 tokens are kept as-is (single chunk, zero overhead).
            chunk_size = 64
            overlap = chunk_size // 4  # 32 token overlap between adjacent chunks
            step = chunk_size - overlap

            query_chunks: List[List[str]] = []
            for query in queries:
                token_ids = self.tokenizer.encode(query, add_special_tokens=False)
                if len(token_ids) <= chunk_size:
                    query_chunks.append([query])
                else:
                    chunk_stats["num_chunked_queries"] += 1
                    chunks: List[str] = []
                    for s in range(0, len(token_ids), step):
                        chunk_ids = token_ids[s: s + chunk_size]
                        chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                        chunks.append(chunk_text)
                    query_chunks.append(chunks)

            chunk_counts = [len(chunks) for chunks in query_chunks]
            if chunk_counts:
                chunk_stats["total_chunks"] = int(sum(chunk_counts))
                chunk_stats["avg_chunks_per_query"] = float(sum(chunk_counts)) / len(chunk_counts)
                chunk_stats["max_chunks_for_one_query"] = max(chunk_counts)

            # Flatten all chunks into one batch for a single model forward pass.
            flat_chunks: List[str] = []
            chunk_counts = []
            for chunks in query_chunks:
                flat_chunks.extend(chunks)
                chunk_counts.append(len(chunks))

            inputs = self.tokenizer(
                flat_chunks,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            outputs = self.model(**inputs, return_dict=True)
            flat_emb = pooling(
                pooler_output=getattr(outputs, "pooler_output", None),
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=inputs["attention_mask"],
                pooling_method=self.pooling_method,
            )
            flat_emb = torch.nn.functional.normalize(flat_emb, dim=-1)

            # Average chunk embeddings per query, then re-normalize.
            embeddings_list: List[torch.Tensor] = []
            offset = 0
            for count in chunk_counts:
                chunk_embs = flat_emb[offset: offset + count]  # (count, dim)
                avg_emb = chunk_embs.mean(dim=0)               # (dim,)
                avg_emb = torch.nn.functional.normalize(avg_emb, dim=0)
                embeddings_list.append(avg_emb)
                offset += count

            embeddings = torch.stack(embeddings_list, dim=0)   # (N, dim)
            embeddings_t = embeddings
        else:
            # Simple path: tokenize and embed directly (truncation at max_length).
            inputs = self.tokenizer(
                list(queries),
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            outputs = self.model(**inputs, return_dict=True)
            embeddings_t = pooling(
                pooler_output=getattr(outputs, "pooler_output", None),
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=inputs["attention_mask"],
                pooling_method=self.pooling_method,
            )
            embeddings_t = torch.nn.functional.normalize(embeddings_t, dim=-1)

        self.last_chunk_stats = chunk_stats
        if output_target == "gpu":
            if self.device.type != "cuda":
                raise RuntimeError("GPU embedding output requested from a CPU embedding stage.")
            synchronize_cuda_if_needed(self.cuda_device_index)
            return None, embeddings_t.float().contiguous(), time.perf_counter() - start
        if output_target == "cpu":
            if self.device.type == "cuda":
                synchronize_cuda_if_needed(self.cuda_device_index)
            embeddings_np = embeddings_t.detach().cpu().numpy().astype(np.float32, order="C")
            return embeddings_np, None, time.perf_counter() - start
        raise ValueError(f"Unsupported embedding output target: {output_target}")


class RetrievalStage:
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        topk: int,
        nprobe: int,
        backend: str,
        gpu_id: int,
    ) -> None:
        import faiss

        self.faiss = faiss
        self.topk = int(topk)
        self.nprobe = int(nprobe)
        self.backend = backend
        self.cuda_device_index: Optional[int] = None
        self.gpu_resources = None
        self.faiss_torch_interop = False
        self.logger = logging.getLogger("retrieval_stage")

        self.logger.info("Reading FAISS index from: %s ...", index_path)
        cpu_index = self.faiss.read_index(index_path)
        self.logger.info("FAISS index loaded: ntotal=%d, d=%d", cpu_index.ntotal, cpu_index.d)
        if hasattr(cpu_index, "nprobe"):
            cpu_index.nprobe = int(nprobe)

        if backend == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("xR=1 requires CUDA, but CUDA is not available.")
            if not hasattr(self.faiss, "StandardGpuResources"):
                raise RuntimeError("FAISS GPU APIs unavailable. Install faiss-gpu for xR=1.")
            self.cuda_device_index = gpu_id
            self.gpu_resources = self.faiss.StandardGpuResources()
            self.index = self.faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, cpu_index)
            try:
                import faiss.contrib.torch_utils  # noqa: F401

                self.faiss_torch_interop = True
            except Exception:
                self.faiss_torch_interop = False
        elif backend == "cpu":
            self.index = cpu_index
        else:
            raise ValueError(f"Unknown retrieval backend: {backend}")

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = int(nprobe)

        self.logger.info("Loading corpus from: %s ...", corpus_path)
        self.corpus = load_corpus(corpus_path)
        self.logger.info("Corpus loaded: %d documents.", len(self.corpus))

    def __call__(
        self,
        embeddings_cpu: Optional[np.ndarray] = None,
        embeddings_gpu: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[str]], float]:
        start = time.perf_counter()
        if embeddings_cpu is not None and embeddings_gpu is not None:
            raise ValueError("Provide either embeddings_cpu or embeddings_gpu, not both.")

        if self.backend == "gpu":
            if embeddings_gpu is not None:
                if self.faiss_torch_interop:
                    search_input = embeddings_gpu.detach().float().contiguous()
                else:
                    search_input = embeddings_gpu.detach().cpu().numpy().astype(np.float32, order="C")
            elif embeddings_cpu is not None:
                search_input = np.ascontiguousarray(embeddings_cpu.astype(np.float32, copy=False))
            else:
                raise ValueError("GPU retrieval requires embeddings input.")
        else:
            if embeddings_cpu is not None:
                search_input = np.ascontiguousarray(embeddings_cpu.astype(np.float32, copy=False))
            elif embeddings_gpu is not None:
                search_input = (
                    embeddings_gpu.detach().cpu().numpy().astype(np.float32, order="C")
                )
            else:
                raise ValueError("CPU retrieval requires embeddings input.")

        _, indices = self.index.search(search_input, self.topk)
        if self.backend == "gpu":
            synchronize_cuda_if_needed(self.cuda_device_index)
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices)

        retrieved_docs: List[List[str]] = []
        corpus_size = len(self.corpus)
        for row in indices:
            docs: List[str] = []
            for idx in row:
                if idx < 0 or idx >= corpus_size:
                    continue
                doc = self.corpus[int(idx)]
                docs.append(extract_doc_text(doc))
            retrieved_docs.append(docs)

        return retrieved_docs, time.perf_counter() - start


class GenerationStage:
    def __init__(
        self,
        model_path: str,
        prompt_template: str,
        max_output_len: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        max_model_len: Optional[int],
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("xG is fixed to GPU in this pipeline, but CUDA is not available.")

        from vllm import LLM, SamplingParams

        self.prompt_template = prompt_template
        self.max_output_len = max_output_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.SamplingParams = SamplingParams
        self._fallback_tokenizer = None

        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enforce_eager": bool(enforce_eager),
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)

        logger = logging.getLogger("generation_stage")
        logger.info("Initializing vLLM: model=%s, tp=%d, gpu_util=%.2f ...",
                    model_path, int(tensor_parallel_size), float(gpu_memory_utilization))
        self.llm = LLM(**llm_kwargs)
        logger.info("vLLM model loaded successfully.")
        self._fallback_tokenizer = self.llm.get_tokenizer()

    def __call__(
        self,
        queries: Sequence[str],
        retrieved_docs: Sequence[Sequence[str]],
    ) -> Tuple[List[str], float, List[int]]:
        prompts = []
        for query, docs in zip(queries, retrieved_docs):
            context = "\n".join(docs)
            prompts.append(self.prompt_template.format(query=query, context=context))

        sampling_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_len,
        }
        if self.top_k is not None:
            sampling_kwargs["top_k"] = int(self.top_k)
        sampling_params = self.SamplingParams(**sampling_kwargs)

        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        answers: List[str] = []
        generated_tokens: List[int] = []

        for output in outputs:
            if not output.outputs:
                answers.append("")
                generated_tokens.append(0)
                continue
            first_output = output.outputs[0]
            answers.append(first_output.text)
            token_ids = getattr(first_output, "token_ids", None)
            if token_ids is not None:
                generated_tokens.append(len(token_ids))
            else:
                generated_tokens.append(len(self._fallback_tokenizer.encode(first_output.text)))

        return answers, time.perf_counter() - start, generated_tokens


@dataclass
class BatchStats:
    batch_index: int
    batch_size: int
    bucket: str
    embedding_sec: float
    retrieval_sec: float
    generation_sec: float
    generated_tokens: int
    xE: int = 0
    xR: int = 0


@dataclass
class ScheduledMicrobatch:
    bucket: str
    query_indices: List[int]
    queries: List[str]
    token_lengths: List[int]
    action: Dict[str, Any]


@dataclass
class EmbeddingPayload:
    microbatch: ScheduledMicrobatch
    embedding_sec: float
    embeddings_cpu: Optional[np.ndarray]
    embeddings_gpu: Optional[torch.Tensor]


@dataclass
class RetrievalPayload:
    microbatch: ScheduledMicrobatch
    embedding_sec: float
    retrieval_sec: float
    retrieved_docs: List[List[str]]


@dataclass
class PendingQuery:
    query_index: int
    query: str
    token_length: int
    estimated_cost: float


@dataclass
class DispatchTraceEntry:
    dispatch_index: int
    bucket: str
    chosen_batch_size: int
    candidate_batch_sizes: List[int]
    chosen_action: Dict[str, int]
    candidate_actions: List[Dict[str, Any]]
    q_er_len: int
    q_rg_len: int
    predicted_action_cost_ms_per_query: float
    predicted_dispatch_cost_ms_per_query: float
    pending_queries_by_bucket: Dict[str, int]


@dataclass
class FeedbackTraceEntry:
    batch_index: int
    bucket: str
    batch_size: int
    xE: int
    xR: int
    token_length_min: int
    token_length_max: int
    token_length_avg: float
    embedding_ms_per_query: float
    retrieval_ms_per_query: float
    generation_ms_per_query: float
    transfer_ms_per_query_est: float
    batch_size_residual_ms_per_query: float
    ema_after_update: Dict[str, float]


@dataclass
class ChunkTraceEntry:
    batch_index: int
    bucket: str
    num_queries: int
    num_chunked_queries: int
    total_chunks: int
    avg_chunks_per_query: float
    max_chunks_for_one_query: int


@dataclass
class GenerationTargetTraceEntry:
    dispatch_index: int
    target_batch_min: int
    target_batch_ideal: int
    target_batch_max: int
    pending_count: int
    q_er_len: int
    q_rg_len: int
    gpu_free_mem_gb: float


@dataclass
class DevicePlanTraceEntry:
    dispatch_index: int
    chosen_action: Dict[str, Any]
    keep_gpu_resident_er: bool
    expected_supportable_batch_size: int
    predicted_action_cost_ms_per_query: float
    gpu_free_mem_gb: float


@dataclass
class BatchShapingTraceEntry:
    dispatch_index: int
    pre_shape_token_min: int
    pre_shape_token_max: int
    pre_shape_token_avg: float
    post_shape_token_min: int
    post_shape_token_max: int
    post_shape_token_avg: float
    chosen_batch_size: int
    shaping_applied: bool


@dataclass
@dataclass
class ResourceTrackerSnapshot:
    gpu_free_mem_gb: float
    gpu_total_mem_gb: float
    pressure_level: str
    estimated_vllm_kv_gb: float
    estimated_faiss_index_gb: float
    estimated_embed_activations_gb: float
    available_for_new_batch_gb: float


class ResourceTracker:
    """
    Tracks GPU memory state in real time and estimates per-stage memory costs.

    In resource-constrained scenarios, the scheduler makes better decisions when it
    knows how much GPU memory is actually available for the current batch, rather
    than relying on a static threshold.
    """

    # e5-large-v2 embedding activations: ~(bytes_per_token * layers * hidden/2) for fp16
    # Rough model: ~0.4 GiB for model weights + ~0.001 GiB per query in activations
    EMBED_MODEL_WEIGHTS_GB = 0.4
    EMBED_ACTIVATIONS_PER_QUERY_GB = 0.0002  # 0.2 MiB per query (fp16)

    # FAISS index GPU memory: loaded on-demand, typically 1-4 GiB for small corpuses
    FAISS_INDEX_DEFAULT_GB = 2.0

    # vLLM KV cache: roughly proportional to gpu_memory_utilization and max_model_len
    # The actual reserved amount is set by vLLM at init time.
    VLLM_RESERVED_FRACTION = 0.5  # default vLLM gpu_memory_utilization baseline

    def __init__(
        self,
        gpu_id: int,
        vllm_gpu_memory_utilization: float,
        faiss_index_gb: float = FAISS_INDEX_DEFAULT_GB,
        embed_model_weights_gb: float = EMBED_MODEL_WEIGHTS_GB,
    ) -> None:
        self.gpu_id = gpu_id
        self.vllm_reserved_gb = 0.0  # set after vLLM loads
        self.vllm_utilization = vllm_gpu_memory_utilization
        self.faiss_index_gb = faiss_index_gb
        self.embed_model_weights_gb = embed_model_weights_gb

        self._vllm_init_done = False
        self._snapshot_count = 0

    def set_vllm_init_complete(self, gpu_total_gb: float) -> None:
        """Call after vLLM is loaded; records how much GPU RAM vLLM reserved."""
        self.vllm_reserved_gb = gpu_total_gb * self.vllm_utilization
        self._vllm_init_done = True

    def get_current_free_mem_gb(self) -> float:
        """Returns real-time free GPU memory in GiB."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            free, _total = torch.cuda.mem_get_info(self.gpu_id)
            return float(free) / (1024 ** 3)
        except Exception:
            return 0.0

    def get_current_total_mem_gb(self) -> float:
        """Returns total GPU memory in GiB."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            props = torch.cuda.get_device_properties(self.gpu_id)
            return float(props.total_memory) / (1024 ** 3)
        except Exception:
            return 0.0

    def estimate_embed_activations_gb(self, batch_size: int, x_e: int) -> float:
        """Estimated GPU memory consumed by embedding activations for this batch."""
        if x_e == 0:
            return 0.0  # CPU embed, no GPU activations
        return batch_size * self.EMBED_ACTIVATIONS_PER_QUERY_GB

    def estimate_retrieval_cost_gb(self, batch_size: int, x_r: int) -> float:
        """Estimated GPU memory consumed by FAISS GPU retrieval for this batch."""
        if x_r == 0:
            return 0.0  # CPU retrieval, FAISS stays on CPU
        # FAISS GPU index: stays resident while xR=1, shared with vLLM
        return 0.0  # index memory is tracked separately as self.faiss_index_gb

    def estimate_generation_cost_gb(self, batch_size: int, avg_input_tokens: int, max_output_tokens: int) -> float:
        """Rough estimate of additional GPU memory needed for generation of this batch.

        The dominant cost is in vLLM's KV cache, which scales with:
        - batch_size
        - total tokens (input + output)
        - hidden_dim
        For Llama-3.1-8B at fp16: ~0.00004 GiB per token per layer.
        """
        if not self._vllm_init_done:
            return 0.0
        layers = 32  # Llama-3.1-8B
        hidden = 4096
        bytes_per_param = 2.0  # fp16
        tokens_per_sample = avg_input_tokens + max_output_tokens
        # KV cache: 2 * layers * hidden * tokens_per_sample * bytes_per_param
        kv_per_sample = 2 * layers * hidden * bytes_per_param
        return batch_size * tokens_per_sample * kv_per_sample / (1024 ** 3)

    def pressure_level(self, gpu_free_mem_gb: float) -> str:
        """Classifies current memory pressure into three tiers."""
        if gpu_free_mem_gb < 4.0:
            return "high"
        if gpu_free_mem_gb < 10.0:
            return "medium"
        return "low"

    def get_snapshot(self, batch_size: int = 0, x_e: int = 0, x_r: int = 0) -> ResourceTrackerSnapshot:
        """Returns a complete snapshot of memory state, optionally for a pending batch."""
        free = self.get_current_free_mem_gb()
        total = self.get_current_total_mem_gb()
        embed_act_gb = self.estimate_embed_activations_gb(batch_size, x_e)
        available = free - embed_act_gb
        if x_r == 1:
            available -= self.faiss_index_gb

        return ResourceTrackerSnapshot(
            gpu_free_mem_gb=free,
            gpu_total_mem_gb=total,
            pressure_level=self.pressure_level(free),
            estimated_vllm_kv_gb=self.vllm_reserved_gb,
            estimated_faiss_index_gb=(self.faiss_index_gb if x_r == 1 else 0.0),
            estimated_embed_activations_gb=embed_act_gb,
            available_for_new_batch_gb=max(0.0, available),
        )

    def max_batch_size_for_action(
        self,
        x_e: int,
        x_r: int,
        max_theoretical: int = 128,
    ) -> int:
        """Estimates the maximum batch size that fits in GPU memory for a given action.

        This is a conservative estimate used to bound the scheduler's batch size search.
        """
        if not torch.cuda.is_available():
            return 1

        free = self.get_current_free_mem_gb()

        # Memory budget breakdown
        reserved = self.vllm_reserved_gb
        faiss = self.faiss_index_gb if x_r == 1 else 0.0
        embed_weights = self.embed_model_weights_gb if x_e == 1 else 0.0

        available = free - reserved - faiss - embed_weights

        # Generous per-query overhead (embedding activations + retrieval buffers + generation KV)
        per_query_gb = self.EMBED_ACTIVATIONS_PER_QUERY_GB + 0.0005

        max_by_mem = int(available / per_query_gb)
        return max(1, min(max_by_mem, max_theoretical))


class GreedyBucketScheduler:
    def __init__(self, args: argparse.Namespace, resource_tracker: Optional[ResourceTracker] = None) -> None:
        self.args = args
        self.resource_tracker = resource_tracker
        self.scheduler_mode_choice = getattr(args, "scheduler_mode_choice", "legacy_bucket")
        self.short_threshold = int(args.length_short_threshold)
        self.long_threshold = int(args.length_long_threshold)

        self.batch_short = int(args.bucket_batch_short)
        self.batch_mid = int(args.bucket_batch_mid)
        self.batch_long = int(args.bucket_batch_long)

        self.embed_long_gpu_threshold = int(args.embed_long_gpu_threshold)
        self.retrieve_gpu_batch_threshold = int(args.retrieve_gpu_batch_threshold)
        self.backpressure_high = int(args.backpressure_high)
        self.ema_alpha = float(getattr(args, "scheduler_ema_alpha", 0.25))

        # Memory-aware thresholds (GiB) — these replace the static 20 GiB hard cap on xR=1.
        self.gpu_mem_low_threshold_gb = float(getattr(args, "gpu_mem_low_threshold_gb", 4.0))
        self.gpu_mem_medium_threshold_gb = float(getattr(args, "gpu_mem_medium_threshold_gb", 10.0))
        self.gpu_mem_high_batch_penalty = float(getattr(args, "gpu_mem_high_batch_penalty", 50.0))
        self.enable_memory_aware_scheduling = getattr(args, "enable_memory_aware_scheduling", True)

        # async_bucket 模式下调度器拥有完整 action 空间。
        self.available_actions = [
            {"xE": 0, "xR": 0},
            {"xE": 1, "xR": 0},
            {"xE": 0, "xR": 1},
            {"xE": 1, "xR": 1},
        ]
        self._embedding_latency_ema_ms_per_query: Dict[Tuple[str, int], float] = {}
        self._retrieval_latency_ema_ms_per_query: Dict[Tuple[str, int], float] = {}
        self._generation_latency_ema_ms_per_query: Dict[str, float] = {}
        self._transfer_latency_ema_ms_per_query: Dict[Tuple[str, int, int], float] = {}
        self._batch_size_residual_ema_ms_per_query: Dict[Tuple[str, int], float] = {}
        self._bucket_batch_size_ema: Dict[str, float] = {
            "short": float(self.batch_short),
            "mid": float(self.batch_mid),
            "long": float(self.batch_long),
        }
        self._pending_by_bucket: Dict[str, List[PendingQuery]] = {"short": [], "mid": [], "long": []}
        self._pending_queries: List[PendingQuery] = []
        self._query_token_lengths: Dict[int, int] = {}
        self.dispatch_trace: List[DispatchTraceEntry] = []
        self.feedback_trace: List[FeedbackTraceEntry] = []
        self.generation_target_trace: List[GenerationTargetTraceEntry] = []
        self.device_plan_trace: List[DevicePlanTraceEntry] = []
        self.batch_shaping_trace: List[BatchShapingTraceEntry] = []

    @staticmethod
    def _estimate_query_length(query: str) -> int:
        return max(1, len(query.split()))

    def _bucket_name(self, length: int) -> str:
        if length <= self.short_threshold:
            return "short"
        if length <= self.long_threshold:
            return "mid"
        return "long"

    def _bucket_batch_size(self, bucket: str) -> int:
        return {
            "short": self.batch_short,
            "mid": self.batch_mid,
            "long": self.batch_long,
        }[bucket]

    def _candidate_batch_sizes(self, bucket: str, available: int) -> List[int]:
        if getattr(self.args, "ablate_online_batch", False):
            return [min(self._bucket_batch_size(bucket), available)]
        base = self._bucket_batch_size(bucket)
        ema_size = max(1, int(round(self._bucket_batch_size_ema[bucket])))
        candidates = {
            max(1, base // 2),
            base,
            min(base * 2, max(base, ema_size)),
            ema_size,
            max(1, ema_size // 2),
        }
        ordered = sorted({size for size in candidates if size > 0})
        return [min(size, available) for size in ordered if min(size, available) > 0]

    def _generation_target_bounds(
        self,
        pending_count: int,
        gpu_free_mem_gb: float,
        q_er_len: int,
        q_rg_len: int,
    ) -> Tuple[int, int, int]:
        if pending_count <= 0:
            return 1, 1, 1
        base_ideal = 64
        if gpu_free_mem_gb < 12.0:
            base_ideal = 16
        elif gpu_free_mem_gb < 18.0:
            base_ideal = 32
        if q_rg_len >= self.backpressure_high:
            base_ideal = max(8, base_ideal // 2)
        if q_er_len >= self.backpressure_high:
            base_ideal = max(8, base_ideal // 2)
        ideal = min(base_ideal, pending_count)
        target_min = max(4, ideal // 2)
        target_max = min(max(ideal, target_min), pending_count)
        return target_min, ideal, target_max

    def _candidate_generation_batch_sizes(
        self,
        pending_count: int,
        target_min: int,
        target_ideal: int,
        target_max: int,
    ) -> List[int]:
        candidates = {
            target_min,
            target_ideal,
            target_max,
            max(4, target_ideal // 2),
            min(pending_count, target_ideal + max(4, target_ideal // 2)),
        }
        ordered = sorted({min(size, pending_count) for size in candidates if size > 0})
        return [size for size in ordered if size > 0]

    def _plan_device_for_batch(
        self,
        token_lengths: Sequence[int],
        batch_size: int,
        gpu_available: bool,
        gpu_free_mem_gb: float,
        q_er_len: int,
        q_rg_len: int,
    ) -> Dict[str, Any]:
        action = self._choose_action_for_batch(
            bucket=self._bucket_name(max(token_lengths) if token_lengths else 1),
            lengths=token_lengths,
            batch_size=batch_size,
            gpu_available=gpu_available,
            gpu_mem_gb=gpu_free_mem_gb,
            q_er_len=q_er_len,
            q_rg_len=q_rg_len,
        )
        x_e = int(action["xE"])
        x_r = int(action["xR"])
        keep_gpu_resident_er = x_e == 1 and x_r == 1 and gpu_available
        return {
            "xE": x_e,
            "xR": x_r,
            "keep_gpu_resident_er": keep_gpu_resident_er,
        }

    def _shape_batch(
        self,
        candidate_items: List[PendingQuery],
        target_batch_size: int,
    ) -> List[PendingQuery]:
        if not getattr(self.args, "enable_batch_shaping", False):
            return candidate_items[:target_batch_size]
        if len(candidate_items) <= target_batch_size:
            return candidate_items
        sorted_items = sorted(candidate_items, key=lambda item: item.token_length)
        return sorted_items[:target_batch_size]

    def _bucket_priority(
        self,
        bucket: str,
        waiting_count: int,
        q_er_len: int,
        q_rg_len: int,
        long_ratio: float,
        gpu_mem_gb: float = 999.0,
    ) -> float:
        """
        Memory-pressure-aware bucket priority.

        When GPU memory is tight, we want to:
        - Prioritize 'long' queries (biggest memory footprint) first, so they
          are released early and free up memory for subsequent batches.
        - Prioritize 'short' queries when memory is abundant, since they pack
          well into large batches for generation throughput.
        - 'mid' queries are always a reasonable middle ground.
        """
        base = {"short": 3.0, "mid": 2.0, "long": 1.0}[bucket]
        pressure = waiting_count * 0.05 + q_er_len * 0.04 + q_rg_len * 0.06
        if bucket == "long":
            pressure += long_ratio

        # Memory pressure bonus: in tight memory, long queries deserve higher priority
        # because they consume the most memory and should be dispatched early to be
        # released early (especially important when xE=1/xR=1 is constrained).
        if self.enable_memory_aware_scheduling and self.resource_tracker is not None:
            mem_pressure = self.resource_tracker.pressure_level(gpu_mem_gb)
            if mem_pressure == "high":
                # Long queries take the most GPU memory per query; dispatch them first
                # to avoid being stuck in pending queue when memory runs out.
                if bucket == "long":
                    base += 3.0
                elif bucket == "mid":
                    base += 1.5
                elif bucket == "short":
                    base -= 0.5  # de-prioritize: short queries are cheap, can wait
            elif mem_pressure == "medium":
                if bucket == "long":
                    base += 1.0
                elif bucket == "short":
                    base -= 0.5

        return base + pressure

    @staticmethod
    def _estimate_embedding_cost(length: int) -> float:
        return 0.0074 * (length ** 2) + 0.17 * length

    def _estimate_generation_cost(self, bucket: str, lengths: Sequence[int]) -> float:
        l_max = max(lengths) if lengths else 1
        base = {"short": 6.0, "mid": 8.5, "long": 12.0}[bucket]
        return base + 0.03 * l_max

    def _estimate_transfer_cost_for_action(self, bucket: str, x_e: int, x_r: int) -> float:
        observed = self._transfer_latency_ema_ms_per_query.get((bucket, x_e, x_r))
        if observed is not None:
            return observed
        if x_e == 1 and x_r == 0:
            return 1.0
        if x_e == 0 and x_r == 1:
            return 1.2
        return 0.1

    def _estimate_embedding_cost_for_action(
        self,
        bucket: str,
        lengths: Sequence[int],
        x_e: int,
    ) -> float:
        observed = self._embedding_latency_ema_ms_per_query.get((bucket, x_e))
        if observed is not None:
            return observed
        l_max = max(lengths) if lengths else 1
        if x_e == 0:
            return self._estimate_embedding_cost(l_max)
        return 0.00015 * (l_max ** 2) + 0.008 * l_max + 0.5

    def _estimate_retrieval_cost_for_action(
        self,
        bucket: str,
        batch_size: int,
        x_r: int,
    ) -> float:
        observed = self._retrieval_latency_ema_ms_per_query.get((bucket, x_r))
        if observed is not None:
            return observed
        return (4.5 if x_r == 0 else 1.2) + 8.0 / max(1, batch_size)

    def _estimate_generation_cost_for_bucket(
        self,
        bucket: str,
        lengths: Sequence[int],
    ) -> float:
        observed = self._generation_latency_ema_ms_per_query.get(bucket)
        if observed is not None:
            return observed
        return self._estimate_generation_cost(bucket, lengths)

    def _estimate_batch_size_residual(self, bucket: str, batch_size: int) -> float:
        observed = self._batch_size_residual_ema_ms_per_query.get((bucket, batch_size))
        if observed is not None:
            return observed
        return 6.0 / max(1, batch_size)

    def _estimate_action_cost(
        self,
        bucket: str,
        lengths: Sequence[int],
        batch_size: int,
        x_e: int,
        x_r: int,
    ) -> float:
        emb_cost = self._estimate_embedding_cost_for_action(bucket, lengths, x_e)
        ret_cost = self._estimate_retrieval_cost_for_action(bucket, batch_size, x_r)
        transfer_cost = self._estimate_transfer_cost_for_action(bucket, x_e, x_r)
        return emb_cost + ret_cost + transfer_cost

    def _estimate_dispatch_cost(
        self,
        bucket: str,
        lengths: Sequence[int],
        batch_size: int,
        x_e: int,
        x_r: int,
    ) -> float:
        return (
            self._estimate_action_cost(bucket, lengths, batch_size, x_e, x_r)
            + self._estimate_generation_cost_for_bucket(bucket, lengths)
            + self._estimate_batch_size_residual(bucket, batch_size)
        )

    def _compute_overlap_potential(
        self,
        x_e: int,
        x_r: int,
        gpu_mem_gb: float,
    ) -> float:
        """
        Estimates how much pipeline overlap is possible for a given action.

        Returns a score from 0.0 to 1.0:
        - 1.0 = full overlap possible (CPU embed + GPU retrieve/generate can run in parallel)
        - 0.0 = no overlap (all stages on GPU, GPU is saturated)

        When xE=0 (CPU embed), the embedding runs on CPU and can fully overlap with
        GPU-based retrieval and generation. When xE=1 and xR=0, embeddings stay on GPU
        but compete with retrieval/generation, so overlap is reduced.

        In resource-constrained scenarios, maximizing overlap potential is critical
        because CPU embed + GPU retrieve is the key strategy to keep GPU utilization high.
        """
        if not self.enable_memory_aware_scheduling or self.resource_tracker is None:
            return 0.5  # neutral default

        pressure = self.resource_tracker.pressure_level(gpu_mem_gb)

        if pressure == "high":
            # In high memory pressure, xE=0 (CPU embed) is the best choice because it
            # frees GPU entirely for retrieval + generation. Overlap potential is maximum.
            if x_e == 0:
                return 1.0
            # xE=1 + xR=0: embedding competes with retrieval for GPU — overlap is poor
            if x_e == 1 and x_r == 0:
                return 0.2
            return 0.0

        if pressure == "medium":
            if x_e == 0:
                return 0.9
            if x_e == 1 and x_r == 0:
                return 0.4
            return 0.3

        # Low pressure: GPU has headroom, overlap is less critical
        if x_e == 0:
            return 0.7
        return 0.3

    def _record_batch_feedback(
        self,
        bucket: str,
        x_e: int,
        x_r: int,
        batch_size: int,
        token_lengths: Sequence[int],
        embedding_sec: float,
        retrieval_sec: float,
        generation_sec: float,
    ) -> None:
        predicted_base = (
            self._estimate_action_cost(bucket, token_lengths, batch_size, x_e, x_r)
            + self._estimate_generation_cost_for_bucket(bucket, token_lengths)
        )

        embedding_ms_per_query = embedding_sec * 1000.0 / max(1, batch_size)
        emb_key = (bucket, x_e)
        previous_emb = self._embedding_latency_ema_ms_per_query.get(emb_key)
        if previous_emb is None:
            self._embedding_latency_ema_ms_per_query[emb_key] = embedding_ms_per_query
        else:
            self._embedding_latency_ema_ms_per_query[emb_key] = (
                self.ema_alpha * embedding_ms_per_query + (1.0 - self.ema_alpha) * previous_emb
            )

        retrieval_ms_per_query = retrieval_sec * 1000.0 / max(1, batch_size)
        ret_key = (bucket, x_r)
        previous_ret = self._retrieval_latency_ema_ms_per_query.get(ret_key)
        if previous_ret is None:
            self._retrieval_latency_ema_ms_per_query[ret_key] = retrieval_ms_per_query
        else:
            self._retrieval_latency_ema_ms_per_query[ret_key] = (
                self.ema_alpha * retrieval_ms_per_query + (1.0 - self.ema_alpha) * previous_ret
            )

        transfer_ms_per_query = 0.0
        if x_e == 1 and x_r == 0:
            transfer_ms_per_query = embedding_ms_per_query * 0.15
        elif x_e == 0 and x_r == 1:
            transfer_ms_per_query = retrieval_ms_per_query * 0.15
        transfer_key = (bucket, x_e, x_r)
        previous_transfer = self._transfer_latency_ema_ms_per_query.get(transfer_key)
        if previous_transfer is None:
            self._transfer_latency_ema_ms_per_query[transfer_key] = transfer_ms_per_query
        else:
            self._transfer_latency_ema_ms_per_query[transfer_key] = (
                self.ema_alpha * transfer_ms_per_query + (1.0 - self.ema_alpha) * previous_transfer
            )

        generation_ms_per_query = generation_sec * 1000.0 / max(1, batch_size)
        previous_gen = self._generation_latency_ema_ms_per_query.get(bucket)
        if previous_gen is None:
            self._generation_latency_ema_ms_per_query[bucket] = generation_ms_per_query
        else:
            self._generation_latency_ema_ms_per_query[bucket] = (
                self.ema_alpha * generation_ms_per_query + (1.0 - self.ema_alpha) * previous_gen
            )

        total_ms_per_query = (embedding_sec + retrieval_sec + generation_sec) * 1000.0 / max(1, batch_size)
        batch_residual = total_ms_per_query - predicted_base
        batch_key = (bucket, batch_size)
        previous_batch = self._batch_size_residual_ema_ms_per_query.get(batch_key)
        if previous_batch is None:
            self._batch_size_residual_ema_ms_per_query[batch_key] = batch_residual
        else:
            self._batch_size_residual_ema_ms_per_query[batch_key] = (
                self.ema_alpha * batch_residual + (1.0 - self.ema_alpha) * previous_batch
            )

        target_batch_size = self._bucket_batch_size_ema[bucket]
        if total_ms_per_query > 0:
            if batch_size <= target_batch_size:
                proposed = min(float(batch_size + 4), target_batch_size + 8.0)
            else:
                proposed = max(4.0, float(batch_size - 4))
            self._bucket_batch_size_ema[bucket] = (
                self.ema_alpha * proposed + (1.0 - self.ema_alpha) * target_batch_size
            )

        self.feedback_trace.append(
            FeedbackTraceEntry(
                batch_index=len(self.feedback_trace) + 1,
                bucket=bucket,
                batch_size=batch_size,
                xE=x_e,
                xR=x_r,
                token_length_min=min(token_lengths) if token_lengths else 0,
                token_length_max=max(token_lengths) if token_lengths else 0,
                token_length_avg=(float(sum(token_lengths)) / len(token_lengths)) if token_lengths else 0.0,
                embedding_ms_per_query=embedding_ms_per_query,
                retrieval_ms_per_query=retrieval_ms_per_query,
                generation_ms_per_query=generation_ms_per_query,
                transfer_ms_per_query_est=self._transfer_latency_ema_ms_per_query.get(transfer_key, 0.0),
                batch_size_residual_ms_per_query=self._batch_size_residual_ema_ms_per_query.get(batch_key, 0.0),
                ema_after_update={
                    "embedding": self._embedding_latency_ema_ms_per_query.get(emb_key, 0.0),
                    "retrieval": self._retrieval_latency_ema_ms_per_query.get(ret_key, 0.0),
                    "generation": self._generation_latency_ema_ms_per_query.get(bucket, 0.0),
                    "transfer": self._transfer_latency_ema_ms_per_query.get(transfer_key, 0.0),
                    "batch_size_residual": self._batch_size_residual_ema_ms_per_query.get(batch_key, 0.0),
                    "bucket_batch_size_ema": self._bucket_batch_size_ema.get(bucket, 0.0),
                },
            )
        )

    def _pop_batch_queries(self, bucket: str, batch_size: int) -> List[PendingQuery]:
        pending = self._pending_by_bucket[bucket]
        if not pending:
            return []
        take = min(batch_size, len(pending))
        batch = pending[:take]
        del pending[:take]
        return batch

    def _action_feasible(
        self,
        x_e: int,
        x_r: int,
        batch_size: int,
        gpu_available: bool,
        gpu_mem_gb: float,
    ) -> Tuple[bool, float]:
        """
        Checks if an (xE, xR, batch_size) combination is feasible under current GPU memory.

        Returns (feasible, memory_penalty).
        The memory_penalty is a rough estimate of the additional cost if this action
        would push the GPU into a tighter memory regime.
        """
        if (x_e == 1 or x_r == 1) and not gpu_available:
            return False, 0.0

        if not self.enable_memory_aware_scheduling:
            # Fallback to old static threshold behavior
            if x_r == 1 and gpu_mem_gb < 20.0:
                return False, 0.0
            if x_r == 1 and batch_size < self.retrieve_gpu_batch_threshold:
                return False, 0.0
            return True, 0.0

        # --- Memory-aware feasibility check ---
        if self.resource_tracker is not None:
            max_feasible = self.resource_tracker.max_batch_size_for_action(x_e, x_r)
            if batch_size > max_feasible:
                return False, 0.0

        pressure = self.resource_tracker.pressure_level(gpu_mem_gb) if self.resource_tracker else "low"

        # Memory pressure penalties: the tighter the memory, the more we penalize
        # actions that consume GPU resources heavily.
        memory_penalty = 0.0
        if pressure == "high":
            if x_r == 1:
                memory_penalty += self.gpu_mem_high_batch_penalty
            elif x_e == 1:
                memory_penalty += self.gpu_mem_high_batch_penalty * 0.5
        elif pressure == "medium":
            if x_r == 1 and gpu_mem_gb < self.gpu_mem_medium_threshold_gb:
                memory_penalty += 10.0
            if x_e == 1 and gpu_mem_gb < self.gpu_mem_low_threshold_gb:
                memory_penalty += 5.0

        return True, memory_penalty

    def _choose_action_for_batch(
        self,
        bucket: str,
        lengths: Sequence[int],
        batch_size: int,
        gpu_available: bool,
        gpu_mem_gb: float,
        q_er_len: int,
        q_rg_len: int,
    ) -> Dict[str, Any]:
        if getattr(self.args, "ablate_online_action", False):
            return {"xE": int(self.args.xE), "xR": int(self.args.xR)}
        l_max = max(lengths) if lengths else 1

        candidates = []
        for action in self.available_actions:
            x_e = int(action["xE"])
            x_r = int(action["xR"])

            feasible, memory_penalty = self._action_feasible(
                x_e, x_r, batch_size, gpu_available, gpu_mem_gb
            )
            if not feasible:
                continue

            score = self._estimate_action_cost(bucket, lengths, batch_size, x_e, x_r)
            if x_e == 0:
                score += q_er_len * 0.03
            if x_r == 0:
                score += q_rg_len * 0.05
            if l_max >= self.embed_long_gpu_threshold and x_e == 0:
                score += 20.0
            if l_max >= self.long_threshold and x_e == 1:
                score -= 0.5
            score += memory_penalty
            candidates.append((score, {"xE": x_e, "xR": x_r}))

        if not candidates:
            return {"xE": 0, "xR": 0}

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def prepare_queries(
        self,
        queries: Sequence[str],
        token_lengths: Sequence[int],
    ) -> float:
        self.dispatch_trace = []
        self.feedback_trace = []
        self.generation_target_trace = []
        self.device_plan_trace = []
        self.batch_shaping_trace = []
        self._query_token_lengths = {idx: token_length for idx, token_length in enumerate(token_lengths)}

        if self.scheduler_mode_choice == "generation_target_v1":
            pending = [
                PendingQuery(
                    query_index=idx,
                    query=query,
                    token_length=token_length,
                    estimated_cost=self._estimate_embedding_cost(token_length),
                )
                for idx, (query, token_length) in enumerate(zip(queries, token_lengths))
            ]
            pending.sort(key=lambda item: item.query_index)
            self._pending_queries = pending
            self._pending_by_bucket = {"short": [], "mid": [], "long": []}
            long_count = sum(1 for token_length in token_lengths if token_length > self.long_threshold)
            return long_count / max(1, len(queries))

        if getattr(self.args, "ablate_bucketing", False):
            self._pending_by_bucket = {"short": [], "mid": [], "long": []}
            pending = [
                PendingQuery(
                    query_index=idx,
                    query=query,
                    token_length=token_length,
                    estimated_cost=self._estimate_embedding_cost(token_length),
                )
                for idx, (query, token_length) in enumerate(zip(queries, token_lengths))
            ]
            pending.sort(key=lambda item: item.estimated_cost, reverse=True)
            self._pending_by_bucket["mid"] = pending
            self._pending_queries = []
            return 0.0
        self._pending_by_bucket = {"short": [], "mid": [], "long": []}
        self._pending_queries = []
        total_queries = max(1, len(queries))
        long_count = 0
        for idx, (query, token_length) in enumerate(zip(queries, token_lengths)):
            bucket = self._bucket_name(token_length)
            if bucket == "long":
                long_count += 1
            self._pending_by_bucket[bucket].append(
                PendingQuery(
                    query_index=idx,
                    query=query,
                    token_length=token_length,
                    estimated_cost=self._estimate_embedding_cost(token_length),
                )
            )
        for bucket in ("short", "mid", "long"):
            self._pending_by_bucket[bucket].sort(key=lambda item: item.estimated_cost, reverse=True)
        return long_count / total_queries

    def has_pending(self) -> bool:
        if self.scheduler_mode_choice == "generation_target_v1":
            return bool(self._pending_queries)
        return any(self._pending_by_bucket[bucket] for bucket in ("short", "mid", "long"))

    def next_dispatch(
        self,
        gpu_available: bool,
        gpu_mem_gb: float,
        q_er_len: int,
        q_rg_len: int,
        long_ratio: float,
    ) -> Optional[ScheduledMicrobatch]:
        if self.scheduler_mode_choice == "generation_target_v1":
            pending_count = len(self._pending_queries)
            if pending_count == 0:
                return None

            target_min, target_ideal, target_max = self._generation_target_bounds(
                pending_count=pending_count,
                gpu_free_mem_gb=gpu_mem_gb,
                q_er_len=q_er_len,
                q_rg_len=q_rg_len,
            )
            self.generation_target_trace.append(
                GenerationTargetTraceEntry(
                    dispatch_index=len(self.generation_target_trace) + 1,
                    target_batch_min=target_min,
                    target_batch_ideal=target_ideal,
                    target_batch_max=target_max,
                    pending_count=pending_count,
                    q_er_len=q_er_len,
                    q_rg_len=q_rg_len,
                    gpu_free_mem_gb=gpu_mem_gb,
                )
            )

            best_batch: Optional[ScheduledMicrobatch] = None
            best_score = float("inf")
            best_device_plan: Optional[Dict[str, Any]] = None
            best_pre_lengths: List[int] = []
            best_post_lengths: List[int] = []
            candidate_batch_sizes = self._candidate_generation_batch_sizes(
                pending_count=pending_count,
                target_min=target_min,
                target_ideal=target_ideal,
                target_max=target_max,
            )
            for batch_size in candidate_batch_sizes:
                candidate_items = self._pending_queries[:batch_size]
                pre_lengths = [item.token_length for item in candidate_items]
                shaped_items = self._shape_batch(candidate_items, batch_size)
                post_lengths = [item.token_length for item in shaped_items]
                device_plan = self._plan_device_for_batch(
                    token_lengths=post_lengths,
                    batch_size=len(shaped_items),
                    gpu_available=gpu_available,
                    gpu_free_mem_gb=gpu_mem_gb,
                    q_er_len=q_er_len,
                    q_rg_len=q_rg_len,
                )
                dispatch_cost = self._estimate_dispatch_cost(
                    bucket=self._bucket_name(max(post_lengths) if post_lengths else 1),
                    lengths=post_lengths,
                    batch_size=len(shaped_items),
                    x_e=int(device_plan["xE"]),
                    x_r=int(device_plan["xR"]),
                )
                if dispatch_cost < best_score:
                    best_score = dispatch_cost
                    best_device_plan = device_plan
                    best_pre_lengths = pre_lengths
                    best_post_lengths = post_lengths
                    best_batch = ScheduledMicrobatch(
                        bucket=self._bucket_name(max(post_lengths) if post_lengths else 1),
                        query_indices=[item.query_index for item in shaped_items],
                        queries=[item.query for item in shaped_items],
                        token_lengths=post_lengths,
                        action={
                            "xE": int(device_plan["xE"]),
                            "xR": int(device_plan["xR"]),
                            "keep_gpu_resident_er": bool(device_plan["keep_gpu_resident_er"]),
                        },
                    )
            if best_batch is None or best_device_plan is None:
                return None

            selected = set(best_batch.query_indices)
            self._pending_queries = [item for item in self._pending_queries if item.query_index not in selected]
            self.device_plan_trace.append(
                DevicePlanTraceEntry(
                    dispatch_index=len(self.device_plan_trace) + 1,
                    chosen_action={
                        "xE": int(best_device_plan["xE"]),
                        "xR": int(best_device_plan["xR"]),
                        "keep_gpu_resident_er": bool(best_device_plan["keep_gpu_resident_er"]),
                    },
                    keep_gpu_resident_er=bool(best_device_plan["keep_gpu_resident_er"]),
                    expected_supportable_batch_size=len(best_batch.queries),
                    predicted_action_cost_ms_per_query=self._estimate_action_cost(
                        bucket=best_batch.bucket,
                        lengths=best_batch.token_lengths,
                        batch_size=len(best_batch.queries),
                        x_e=int(best_device_plan["xE"]),
                        x_r=int(best_device_plan["xR"]),
                    ),
                    gpu_free_mem_gb=gpu_mem_gb,
                )
            )
            self.batch_shaping_trace.append(
                BatchShapingTraceEntry(
                    dispatch_index=len(self.batch_shaping_trace) + 1,
                    pre_shape_token_min=min(best_pre_lengths) if best_pre_lengths else 0,
                    pre_shape_token_max=max(best_pre_lengths) if best_pre_lengths else 0,
                    pre_shape_token_avg=(sum(best_pre_lengths) / len(best_pre_lengths)) if best_pre_lengths else 0.0,
                    post_shape_token_min=min(best_post_lengths) if best_post_lengths else 0,
                    post_shape_token_max=max(best_post_lengths) if best_post_lengths else 0,
                    post_shape_token_avg=(sum(best_post_lengths) / len(best_post_lengths)) if best_post_lengths else 0.0,
                    chosen_batch_size=len(best_batch.queries),
                    shaping_applied=bool(getattr(self.args, "enable_batch_shaping", False)),
                )
            )
            return best_batch

        total_pending = sum(len(self._pending_by_bucket[b]) for b in ("short", "mid", "long"))
        long_ratio = len(self._pending_by_bucket["long"]) / max(1, total_pending)
        best_bucket = None
        best_score = -float("inf")
        best_action: Dict[str, int] = {"xE": 0, "xR": 0}
        best_batch_size = 0
        best_predicted_action_cost = 0.0
        best_predicted_dispatch_cost = 0.0
        best_candidate_batch_sizes: List[int] = []
        best_candidate_actions: List[Dict[str, Any]] = []
        for bucket in ("short", "mid", "long"):
            waiting_queries = len(self._pending_by_bucket[bucket])
            if waiting_queries == 0:
                continue
            waiting_batches = max(1, math.ceil(waiting_queries / max(1, self._bucket_batch_size(bucket))))
            candidate_batch_sizes = self._candidate_batch_sizes(bucket, waiting_queries)
            for batch_size in candidate_batch_sizes:
                candidate_items = self._pending_by_bucket[bucket][:batch_size]
                lengths = [item.token_length for item in candidate_items]
                action = self._choose_action_for_batch(
                    bucket=bucket,
                    lengths=lengths,
                    batch_size=batch_size,
                    gpu_available=gpu_available,
                    gpu_mem_gb=gpu_mem_gb,
                    q_er_len=q_er_len,
                    q_rg_len=q_rg_len,
                )
                candidate_action_rows: List[Dict[str, Any]] = []
                for candidate_action in self.available_actions:
                    cx_e = int(candidate_action["xE"])
                    cx_r = int(candidate_action["xR"])
                    feasible, _ = self._action_feasible(cx_e, cx_r, batch_size, gpu_available, gpu_mem_gb)
                    if not feasible:
                        continue
                    candidate_action_rows.append(
                        {
                            "xE": cx_e,
                            "xR": cx_r,
                            "predicted_action_cost_ms_per_query": self._estimate_action_cost(
                                bucket, lengths, batch_size, cx_e, cx_r
                            ),
                            "predicted_dispatch_cost_ms_per_query": self._estimate_dispatch_cost(
                                bucket, lengths, batch_size, cx_e, cx_r
                            ),
                        }
                    )
                predicted_action_cost = self._estimate_action_cost(
                    bucket=bucket,
                    lengths=lengths,
                    batch_size=batch_size,
                    x_e=int(action["xE"]),
                    x_r=int(action["xR"]),
                )
                dispatch_ms = self._estimate_dispatch_cost(
                    bucket=bucket,
                    lengths=lengths,
                    batch_size=batch_size,
                    x_e=int(action["xE"]),
                    x_r=int(action["xR"]),
                )
                score = self._bucket_priority(bucket, waiting_batches, q_er_len, q_rg_len, long_ratio, gpu_mem_gb)
                score -= dispatch_ms / 50.0
                if score > best_score:
                    best_score = score
                    best_bucket = bucket
                    best_action = {"xE": int(action["xE"]), "xR": int(action["xR"])}
                    best_batch_size = batch_size
                    best_predicted_action_cost = predicted_action_cost
                    best_predicted_dispatch_cost = dispatch_ms
                    best_candidate_batch_sizes = candidate_batch_sizes
                    best_candidate_actions = candidate_action_rows
        if best_bucket is None or best_batch_size <= 0:
            return None
        batch_items = self._pop_batch_queries(best_bucket, best_batch_size)
        self.dispatch_trace.append(
            DispatchTraceEntry(
                dispatch_index=len(self.dispatch_trace) + 1,
                bucket=best_bucket,
                chosen_batch_size=best_batch_size,
                candidate_batch_sizes=best_candidate_batch_sizes,
                chosen_action=best_action,
                candidate_actions=best_candidate_actions,
                q_er_len=q_er_len,
                q_rg_len=q_rg_len,
                predicted_action_cost_ms_per_query=best_predicted_action_cost,
                predicted_dispatch_cost_ms_per_query=best_predicted_dispatch_cost,
                pending_queries_by_bucket={
                    bucket_name: len(self._pending_by_bucket[bucket_name])
                    for bucket_name in ("short", "mid", "long")
                },
            )
        )
        return ScheduledMicrobatch(
            bucket=best_bucket,
            query_indices=[item.query_index for item in batch_items],
            queries=[item.query for item in batch_items],
            token_lengths=[item.token_length for item in batch_items],
            action=best_action,
        )


class StandaloneRAGPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logger = logging.getLogger("standalone_rag_pipeline")
        self.gpu_id = parse_primary_gpu_id(args.gpu_id)
        self.gpu_available = torch.cuda.is_available()
        self.gpu_mem_total_gb = self._detect_gpu_memory_gb()

        # ResourceTracker monitors GPU memory in real time and estimates per-stage costs.
        # In resource-constrained scenarios, this is the foundation for memory-aware scheduling.
        self.resource_tracker = ResourceTracker(
            gpu_id=self.gpu_id,
            vllm_gpu_memory_utilization=args.gpu_memory_utilization,
            faiss_index_gb=float(getattr(args, "faiss_index_gb", 2.0)),
        )

        final_enable = getattr(args, "enable_memory_aware_scheduling", True)
        self.scheduler = GreedyBucketScheduler(args, resource_tracker=self.resource_tracker)
        # propagate the value to the scheduler (avoids getattr at every dispatch call)
        self.scheduler.enable_memory_aware_scheduling = final_enable
        self.logger.info(
            "Memory-aware scheduling: %s",
            "ENABLED" if final_enable else "DISABLED",
        )

        self.embedding_backend = self._map_binary_backend(args.xE, "xE")
        self.retrieval_backend = self._map_binary_backend(args.xR, "xR")

        self.logger.info("Initializing embedding stage: model=%s, backend=%s ...",
                         args.embedding_model, self.embedding_backend)
        self.embedding_stage = QueryEmbeddingStage(
            model_path=args.embedding_model,
            pooling_method=args.pooling_method,
            max_length=args.embedding_max_length,
            backend=self.embedding_backend,
            use_fp16=args.embedding_use_fp16,
            gpu_id=self.gpu_id,
            chunked_embedding=(args.pipeline_mode == "async_bucket" and not getattr(args, "ablate_chunking", False)),
        )
        self.logger.info("Embedding stage ready.")
        self.logger.info("Initializing retrieval stage: index=%s, backend=%s ...",
                         args.index_path, self.retrieval_backend)
        self.retrieval_stage = RetrievalStage(
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            topk=args.topk,
            nprobe=args.nprobe,
            backend=self.retrieval_backend,
            gpu_id=self.gpu_id,
        )
        self.logger.info("Retrieval stage ready.")
        self.generation_stage = GenerationStage(
            model_path=args.generator_model,
            prompt_template=args.prompt_template,
            max_output_len=args.max_output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.vllm_enforce_eager,
            max_model_len=args.max_model_len,
        )
        # After vLLM is loaded, record how much GPU memory it consumed.
        self.resource_tracker.set_vllm_init_complete(self.gpu_mem_total_gb)
        self.logger.info("vLLM GPU memory reserved: %.1f GiB (utilization=%.2f)",
                         self.resource_tracker.vllm_reserved_gb, args.gpu_memory_utilization)
        self.queries = load_queries(args)
        token_prep_start = time.perf_counter()
        self.query_token_lengths = self._compute_query_token_lengths(self.queries)
        self.token_length_prep_sec = time.perf_counter() - token_prep_start
        self.logger.info(
            "Loaded %d queries | b=%d xE=%d(%s) xR=%d(%s) nprobe=%d topk=%d",
            len(self.queries),
            args.b,
            args.xE,
            self.embedding_backend,
            args.xR,
            self.retrieval_backend,
            args.nprobe,
            args.topk,
        )

    def _detect_gpu_memory_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            local_device_index = self.gpu_id
            device_count = torch.cuda.device_count()
            if local_device_index < 0 or local_device_index >= device_count:
                local_device_index = 0
            props = torch.cuda.get_device_properties(local_device_index)
            return float(props.total_memory) / (1024 ** 3)
        except Exception:
            return 0.0

    def _compute_query_token_lengths(self, queries: Sequence[str]) -> List[int]:
        tokenizer = self.embedding_stage.tokenizer
        lengths: List[int] = []
        chunk_size = 256
        for start in range(0, len(queries), chunk_size):
            chunk = list(queries[start : start + chunk_size])
            encoded = tokenizer(
                chunk,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        return lengths

    def _detect_gpu_free_memory_gb(self) -> float:
        """返回当前 GPU 的可用显存（GiB），用于批次准备阶段的 action 过滤。"""
        if not torch.cuda.is_available():
            return 0.0
        try:
            local_device_index = self.gpu_id
            device_count = torch.cuda.device_count()
            if local_device_index < 0 or local_device_index >= device_count:
                local_device_index = 0
            free, _total = torch.cuda.mem_get_info(local_device_index)
            return float(free) / (1024 ** 3)
        except Exception:
            return 0.0

    def _switch_backends_if_needed(self, x_e: int, x_r: int) -> None:
        target_embedding = self._map_binary_backend(x_e, "xE")
        target_retrieval = self._map_binary_backend(x_r, "xR")
        if target_embedding != self.embedding_backend:
            self.embedding_backend = target_embedding
            self.embedding_stage = QueryEmbeddingStage(
                model_path=self.args.embedding_model,
                pooling_method=self.args.pooling_method,
                max_length=self.args.embedding_max_length,
                backend=self.embedding_backend,
                use_fp16=self.args.embedding_use_fp16,
                gpu_id=self.gpu_id,
                chunked_embedding=(
                    self.args.pipeline_mode == "async_bucket" and not getattr(self.args, "ablate_chunking", False)
                ),
            )
        if target_retrieval != self.retrieval_backend:
            self.retrieval_backend = target_retrieval
            self.retrieval_stage = RetrievalStage(
                index_path=self.args.index_path,
                corpus_path=self.args.corpus_path,
                topk=self.args.topk,
                nprobe=self.args.nprobe,
                backend=self.retrieval_backend,
                gpu_id=self.gpu_id,
            )

    def _build_embedding_stage(self, backend: str) -> QueryEmbeddingStage:
        return QueryEmbeddingStage(
            model_path=self.args.embedding_model,
            pooling_method=self.args.pooling_method,
            max_length=self.args.embedding_max_length,
            backend=backend,
            use_fp16=self.args.embedding_use_fp16,
            gpu_id=self.gpu_id,
            chunked_embedding=(
                self.args.pipeline_mode == "async_bucket" and not getattr(self.args, "ablate_chunking", False)
            ),
        )

    def _build_retrieval_stage(self, backend: str) -> RetrievalStage:
        return RetrievalStage(
            index_path=self.args.index_path,
            corpus_path=self.args.corpus_path,
            topk=self.args.topk,
            nprobe=self.args.nprobe,
            backend=backend,
            gpu_id=self.gpu_id,
        )

    @staticmethod
    def _map_binary_backend(x: int, label: str) -> str:
        if x not in (0, 1):
            raise ValueError(f"{label} must be 0 or 1.")
        if x == 1:
            if not torch.cuda.is_available():
                raise RuntimeError(f"{label}=1 requires CUDA, but CUDA is not available.")
            return "gpu"
        return "cpu"

    def _build_plain_microbatches(self) -> List[ScheduledMicrobatch]:
        batch_size = int(self.args.b)
        if batch_size <= 0:
            raise ValueError("b must be positive.")
        x_e = int(self.args.xE)
        x_r = int(self.args.xR)

        microbatches: List[ScheduledMicrobatch] = []
        for start in range(0, len(self.queries), batch_size):
            batch_queries = self.queries[start : start + batch_size]
            batch_lengths = self.query_token_lengths[start : start + batch_size]
            bucket = self.scheduler._bucket_name(max(batch_lengths) if batch_lengths else 1)
            microbatches.append(
                ScheduledMicrobatch(
                    bucket=bucket,
                    query_indices=list(range(start, start + len(batch_queries))),
                    queries=batch_queries,
                    token_lengths=batch_lengths,
                    action={"xE": x_e, "xR": x_r},
                )
            )
        return microbatches

    def _build_summary(
        self,
        mode: str,
        execution: str,
        total_embedding_sec: float,
        total_retrieval_sec: float,
        total_generation_sec: float,
        total_generated_tokens: int,
        records: List[BatchStats],
        samples: List[Dict[str, str]],
        bucket_counts: Dict[str, int],
        action_counts: Dict[str, int],
        max_q_er: int,
        max_q_rg: int,
        wall_time_sec: float,
        dispatch_trace: List[DispatchTraceEntry],
        feedback_trace: List[FeedbackTraceEntry],
        chunk_trace: List[ChunkTraceEntry],
        scheduler_dispatch_sec_total: float,
        scheduler_feedback_sec_total: float,
        generation_target_trace: List[GenerationTargetTraceEntry],
        device_plan_trace: List[DevicePlanTraceEntry],
        batch_shaping_trace: List[BatchShapingTraceEntry],
    ) -> Dict[str, Any]:
        total_sec = total_embedding_sec + total_retrieval_sec + total_generation_sec
        total_queries = len(self.queries)
        summary = {
            "config": {
                "b": self.args.b,
                "xE": self.args.xE,
                "xR": self.args.xR,
                "xG": 1,
                "nprobe": self.args.nprobe,
                "topk": self.args.topk,
                "embedding_model": self.args.embedding_model,
                "generator_model": self.args.generator_model,
            },
            "scheduler": {
                "mode": mode,
                "bucket_thresholds": {
                    "short": self.args.length_short_threshold,
                    "long": self.args.length_long_threshold,
                },
                "bucket_batch_sizes": {
                    "short": self.args.bucket_batch_short,
                    "mid": self.args.bucket_batch_mid,
                    "long": self.args.bucket_batch_long,
                },
                "bucket_counts": bucket_counts,
                "action_counts": action_counts,
                "max_q_er": max_q_er,
                "max_q_rg": max_q_rg,
                "fixed_nprobe": self.args.nprobe,
                "execution": execution,
            },
            "timing_breakdown": {
                "token_length_prep_ms": self.token_length_prep_sec * 1000.0,
                "scheduler_dispatch_ms_total": scheduler_dispatch_sec_total * 1000.0,
                "scheduler_dispatch_ms_avg": (
                    scheduler_dispatch_sec_total * 1000.0 / len(dispatch_trace) if dispatch_trace else 0.0
                ),
                "scheduler_feedback_ms_total": scheduler_feedback_sec_total * 1000.0,
                "scheduler_feedback_ms_avg": (
                    scheduler_feedback_sec_total * 1000.0 / len(feedback_trace) if feedback_trace else 0.0
                ),
            },
            "num_queries": total_queries,
            "avg_embedding_ms": (total_embedding_sec * 1000.0 / total_queries) if total_queries else 0.0,
            "avg_retrieval_ms": (total_retrieval_sec * 1000.0 / total_queries) if total_queries else 0.0,
            "avg_generation_ms": (total_generation_sec * 1000.0 / total_queries) if total_queries else 0.0,
            "total_ms": total_sec * 1000.0,
            "throughput_qps": (total_queries / total_sec) if total_sec > 0 else float("inf"),
            "wall_time_ms": wall_time_sec * 1000.0,
            "wall_throughput_qps": total_queries / wall_time_sec,
            "total_generated_tokens": total_generated_tokens,
            "generation_ms_per_token": (
                (total_generation_sec * 1000.0 / total_generated_tokens)
                if total_generated_tokens > 0
                else float("inf")
            ),
            "samples": samples,
            "per_batch": [record.__dict__ for record in records],
            "dispatch_trace": [record.__dict__ for record in dispatch_trace],
            "feedback_trace": [record.__dict__ for record in feedback_trace],
            "chunk_trace": [record.__dict__ for record in chunk_trace],
            "generation_target_trace": [record.__dict__ for record in generation_target_trace],
            "device_plan_trace": [record.__dict__ for record in device_plan_trace],
            "batch_shaping_trace": [record.__dict__ for record in batch_shaping_trace],
        }
        return summary

    def _warmup(self) -> None:
        """Run one small batch through E→R→G to warm up CUDA kernels and model caches.
        Results are discarded; this ensures all three stages are in a hot state
        before the timed experiment begins."""
        warmup_queries = self.queries[:min(4, len(self.queries))]
        self.logger.info("Warming up pipeline with %d queries (results discarded)...", len(warmup_queries))
        keep_gpu = self.embedding_backend == "gpu" and self.retrieval_backend == "gpu"
        embeddings_cpu, embeddings_gpu, _ = self.embedding_stage(
            warmup_queries,
            output_target="gpu" if keep_gpu else "cpu",
        )
        retrieved_docs, _ = self.retrieval_stage(
            embeddings_cpu=embeddings_cpu,
            embeddings_gpu=embeddings_gpu,
        )
        self.generation_stage(warmup_queries, retrieved_docs)
        if self.gpu_available:
            torch.cuda.synchronize()
        self.logger.info("Warmup complete.")

    def run(self) -> Dict[str, Any]:
        total_embedding_sec = 0.0
        total_retrieval_sec = 0.0
        total_generation_sec = 0.0
        total_generated_tokens = 0
        records: List[BatchStats] = []
        samples: List[Dict[str, str]] = []
        bucket_counts: Dict[str, int] = {"short": 0, "mid": 0, "long": 0}
        action_counts: Dict[str, int] = {}
        max_q_er = 0
        max_q_rg = 0
        chunk_trace: List[ChunkTraceEntry] = []
        scheduler_dispatch_sec_total = 0.0
        scheduler_feedback_sec_total = 0.0

        pipeline_mode = self.args.pipeline_mode
        use_bucket_dispatch = pipeline_mode == "async_bucket"
        bucket_long_ratio = 0.0
        if use_bucket_dispatch:
            bucket_long_ratio = self.scheduler.prepare_queries(
                queries=self.queries,
                token_lengths=self.query_token_lengths,
            )
            if not self.scheduler.has_pending():
                raise ValueError("Scheduler returned no microbatches.")
            scheduler_mode = "online_dispatch_ema_v1"
            scheduled_batches = []
        else:
            scheduled_batches = self._build_plain_microbatches()
            if not scheduled_batches:
                raise ValueError("Scheduler returned no microbatches.")
            # async_plain needs _pending_by_bucket populated so has_pending() works.
            # Mirror the structure that prepare_queries() builds for async_bucket.
            self.scheduler._pending_by_bucket = {"short": [], "mid": [], "long": []}
            self.scheduler._pending_queries = []
            self.scheduler._query_token_lengths = {}
            for microbatch in scheduled_batches:
                for idx, query, token_length in zip(
                    microbatch.query_indices, microbatch.queries, microbatch.token_lengths
                ):
                    bucket = self.scheduler._bucket_name(token_length)
                    self.scheduler._pending_by_bucket[bucket].append(
                        PendingQuery(
                            query_index=idx,
                            query=query,
                            token_length=token_length,
                            estimated_cost=self.scheduler._estimate_embedding_cost(token_length),
                        )
                    )
                    self.scheduler._query_token_lengths[idx] = token_length
            scheduler_mode = "plain_fixed_batch"

        self._warmup()
        phase_start = time.perf_counter()

        if pipeline_mode == "serial":
            total_batches = len(scheduled_batches)
            pbar = tqdm(total=len(self.queries), unit="q", desc="serial", dynamic_ncols=True)
            for batch_index, microbatch in enumerate(scheduled_batches, start=1):
                action = microbatch.action
                x_e = int(action.get("xE", self.args.xE))
                x_r = int(action.get("xR", self.args.xR))
                self._switch_backends_if_needed(x_e=x_e, x_r=x_r)
                action_key = f"xE{x_e}_xR{x_r}"
                action_counts[action_key] = action_counts.get(action_key, 0) + 1

                keep_gpu = x_e == 1 and x_r == 1
                embeddings_cpu, embeddings_gpu, embedding_sec = self.embedding_stage(
                    microbatch.queries,
                    output_target="gpu" if keep_gpu else "cpu",
                )
                retrieved_docs, retrieval_sec = self.retrieval_stage(
                    embeddings_cpu=embeddings_cpu,
                    embeddings_gpu=embeddings_gpu,
                )
                answers, generation_sec, token_counts = self.generation_stage(microbatch.queries, retrieved_docs)

                total_embedding_sec += embedding_sec
                total_retrieval_sec += retrieval_sec
                total_generation_sec += generation_sec
                batch_tokens = int(sum(token_counts))
                total_generated_tokens += batch_tokens
                bucket_counts[microbatch.bucket] = bucket_counts.get(microbatch.bucket, 0) + 1
                records.append(
                    BatchStats(
                        batch_index=batch_index,
                        batch_size=len(microbatch.queries),
                        bucket=microbatch.bucket,
                        embedding_sec=embedding_sec,
                        retrieval_sec=retrieval_sec,
                        generation_sec=generation_sec,
                        generated_tokens=batch_tokens,
                        xE=int(microbatch.action.get("xE", 0)),
                        xR=int(microbatch.action.get("xR", 0)),
                    )
                )

                pbar.update(len(microbatch.queries))
                pbar.set_postfix(
                    batch=f"{batch_index}/{total_batches}",
                    ret=f"{retrieval_sec*1000:.0f}ms",
                    gen=f"{generation_sec*1000:.0f}ms",
                )
                if len(samples) < self.args.show_samples:
                    for query, docs, answer in zip(microbatch.queries, retrieved_docs, answers):
                        samples.append(
                            {
                                "query": query,
                                "top_doc_snippet": (docs[0][:300] if docs else ""),
                                "answer": answer,
                                "bucket": microbatch.bucket,
                                "action": microbatch.action,
                            }
                        )
                        if len(samples) >= self.args.show_samples:
                            break
                if batch_index % self.args.log_interval == 0:
                    elapsed = time.perf_counter() - phase_start
                    avg_sec = elapsed / batch_index
                    eta_sec = avg_sec * (total_batches - batch_index)
                    done_queries = sum(record.batch_size for record in records)
                    self.logger.info(
                        "[serial] batch %d/%d | queries %d/%d | "
                        "emb=%.1fms ret=%.1fms gen=%.1fms | "
                        "elapsed=%.1fs ETA=%.1fs",
                        batch_index, total_batches,
                        done_queries, len(self.queries),
                        embedding_sec * 1000, retrieval_sec * 1000, generation_sec * 1000,
                        elapsed, eta_sec,
                    )

            pbar.close()
            wall_time_sec = max(1e-9, time.perf_counter() - phase_start)
            return self._build_summary(
                mode=scheduler_mode,
                execution="serial_pipeline",
                total_embedding_sec=total_embedding_sec,
                total_retrieval_sec=total_retrieval_sec,
                total_generation_sec=total_generation_sec,
                total_generated_tokens=total_generated_tokens,
                records=records,
                samples=samples,
                bucket_counts=bucket_counts,
                action_counts=action_counts,
                max_q_er=0,
                max_q_rg=0,
                wall_time_sec=wall_time_sec,
                dispatch_trace=[],
                feedback_trace=[],
                chunk_trace=[],
                scheduler_dispatch_sec_total=0.0,
                scheduler_feedback_sec_total=0.0,
                generation_target_trace=[],
                device_plan_trace=[],
                batch_shaping_trace=[],
            )

        q_er: queue.Queue[Optional[ScheduledMicrobatch]] = queue.Queue(maxsize=max(2, self.args.backpressure_high))
        q_rg: queue.Queue[Optional[EmbeddingPayload]] = queue.Queue(maxsize=max(2, self.args.backpressure_high))
        q_out: queue.Queue[Optional[RetrievalPayload]] = queue.Queue()
        error_queue: queue.Queue[Exception] = queue.Queue()
        stats_lock = threading.Lock()
        pbar = tqdm(total=len(self.queries), unit="q", desc=pipeline_mode, dynamic_ncols=True)
        seen_batches = 0
        q_er_size = 0
        q_rg_size = 0
        q_er_max_seen = 0
        q_rg_max_seen = 0

        embed_batch_counter = [0]

        def embed_worker() -> None:
            nonlocal q_er_size, q_er_max_seen, q_rg_size, q_rg_max_seen, scheduler_dispatch_sec_total, scheduler_feedback_sec_total
            try:
                stage_cache: Dict[str, QueryEmbeddingStage] = {self.embedding_backend: self.embedding_stage}
                while True:
                    microbatch = q_er.get()
                    if microbatch is None:
                        q_rg.put(None)
                        break

                    with stats_lock:
                        q_er_size = max(0, q_er_size - 1)

                    action = microbatch.action
                    x_e = int(action.get("xE", self.args.xE))
                    target_backend = self._map_binary_backend(x_e, "xE")
                    local_stage = stage_cache.get(target_backend)
                    if local_stage is None:
                        local_stage = self._build_embedding_stage(target_backend)
                        stage_cache[target_backend] = local_stage

                    keep_gpu = x_e == 1 and int(action.get("xR", self.args.xR)) == 1 and target_backend == "gpu"
                    embeddings_cpu, embeddings_gpu, embedding_sec = local_stage(
                        microbatch.queries,
                        output_target="gpu" if keep_gpu else "cpu",
                    )
                    embed_batch_counter[0] += 1
                    chunk_stats = local_stage.last_chunk_stats or {}
                    chunk_trace.append(
                        ChunkTraceEntry(
                            batch_index=embed_batch_counter[0],
                            bucket=microbatch.bucket,
                            num_queries=int(chunk_stats.get("num_queries", len(microbatch.queries))),
                            num_chunked_queries=int(chunk_stats.get("num_chunked_queries", 0)),
                            total_chunks=int(chunk_stats.get("total_chunks", len(microbatch.queries))),
                            avg_chunks_per_query=float(chunk_stats.get("avg_chunks_per_query", 1.0)),
                            max_chunks_for_one_query=int(chunk_stats.get("max_chunks_for_one_query", 1)),
                        )
                    )
                    self.logger.info(
                        "[embed ] batch %d | size=%d bucket=%s xE=%s | emb=%.1fms | q_rg=%d",
                        embed_batch_counter[0], len(microbatch.queries),
                        microbatch.bucket, target_backend,
                        embedding_sec * 1000, q_rg_size,
                    )
                    payload = EmbeddingPayload(
                        microbatch=microbatch,
                        embedding_sec=embedding_sec,
                        embeddings_cpu=embeddings_cpu,
                        embeddings_gpu=embeddings_gpu,
                    )
                    q_rg.put(payload)
                    with stats_lock:
                        q_rg_size += 1
                        q_rg_max_seen = max(q_rg_max_seen, q_rg_size)
            except Exception as exc:  # pragma: no cover - runtime guard
                error_queue.put(exc)
                q_rg.put(None)

        retrieval_batch_counter = [0]

        def retrieval_worker() -> None:
            nonlocal q_rg_size
            try:
                stage_cache: Dict[str, RetrievalStage] = {self.retrieval_backend: self.retrieval_stage}
                while True:
                    payload = q_rg.get()
                    if payload is None:
                        q_out.put(None)
                        break

                    with stats_lock:
                        q_rg_size = max(0, q_rg_size - 1)

                    action = payload.microbatch.action
                    x_r = int(action.get("xR", self.args.xR))
                    target_backend = self._map_binary_backend(x_r, "xR")
                    local_stage = stage_cache.get(target_backend)
                    if local_stage is None:
                        local_stage = self._build_retrieval_stage(target_backend)
                        stage_cache[target_backend] = local_stage

                    retrieved_docs, retrieval_sec = local_stage(
                        embeddings_cpu=payload.embeddings_cpu,
                        embeddings_gpu=payload.embeddings_gpu,
                    )
                    retrieval_batch_counter[0] += 1
                    self.logger.info(
                        "[retriev] batch %d | size=%d bucket=%s xR=%s | ret=%.1fms",
                        retrieval_batch_counter[0], len(payload.microbatch.queries),
                        payload.microbatch.bucket, target_backend,
                        retrieval_sec * 1000,
                    )
                    q_out.put(
                        RetrievalPayload(
                            microbatch=payload.microbatch,
                            embedding_sec=payload.embedding_sec,
                            retrieval_sec=retrieval_sec,
                            retrieved_docs=retrieved_docs,
                        )
                    )
            except Exception as exc:  # pragma: no cover - runtime guard
                error_queue.put(exc)
                q_out.put(None)

        def generation_worker() -> None:
            nonlocal total_embedding_sec, total_retrieval_sec, total_generation_sec
            nonlocal total_generated_tokens, seen_batches
            nonlocal scheduler_feedback_sec_total
            try:
                while True:
                    payload = q_out.get()
                    if payload is None:
                        break

                    answers, generation_sec, token_counts = self.generation_stage(
                        payload.microbatch.queries, payload.retrieved_docs
                    )
                    batch_tokens = int(sum(token_counts))

                    with stats_lock:
                        seen_batches += 1
                        total_embedding_sec += payload.embedding_sec
                        total_retrieval_sec += payload.retrieval_sec
                        total_generation_sec += generation_sec
                        total_generated_tokens += batch_tokens
                        bucket_counts[payload.microbatch.bucket] = bucket_counts.get(payload.microbatch.bucket, 0) + 1
                        records.append(
                            BatchStats(
                                batch_index=seen_batches,
                                batch_size=len(payload.microbatch.queries),
                                bucket=payload.microbatch.bucket,
                                embedding_sec=payload.embedding_sec,
                                retrieval_sec=payload.retrieval_sec,
                                generation_sec=generation_sec,
                                generated_tokens=batch_tokens,
                                xE=int(payload.microbatch.action.get("xE", 0)),
                                xR=int(payload.microbatch.action.get("xR", 0)),
                            )
                        )
                        feedback_start = time.perf_counter()
                        self.scheduler._record_batch_feedback(
                            bucket=payload.microbatch.bucket,
                            x_e=int(payload.microbatch.action.get("xE", 0)),
                            x_r=int(payload.microbatch.action.get("xR", 0)),
                            batch_size=len(payload.microbatch.queries),
                            token_lengths=payload.microbatch.token_lengths,
                            embedding_sec=payload.embedding_sec,
                            retrieval_sec=payload.retrieval_sec,
                            generation_sec=generation_sec,
                        )
                        scheduler_feedback_sec_total += time.perf_counter() - feedback_start

                    with stats_lock:
                        if len(samples) < self.args.show_samples:
                            for query, docs, answer in zip(
                                payload.microbatch.queries, payload.retrieved_docs, answers
                            ):
                                samples.append(
                                    {
                                        "query": query,
                                        "top_doc_snippet": (docs[0][:300] if docs else ""),
                                        "answer": answer,
                                        "bucket": payload.microbatch.bucket,
                                        "action": payload.microbatch.action,
                                    }
                                )
                                if len(samples) >= self.args.show_samples:
                                    break

                        if seen_batches % self.args.log_interval == 0:
                            elapsed = time.perf_counter() - phase_start
                            done_queries = sum(record.batch_size for record in records)
                            qps = done_queries / elapsed if elapsed > 0 else 0.0
                            ms_per_tok = (
                                total_generation_sec * 1000.0 / total_generated_tokens
                                if total_generated_tokens > 0 else 0.0
                            )
                            pbar.set_postfix(
                                ret=f"{payload.retrieval_sec*1000:.0f}ms",
                                gen=f"{generation_sec*1000:.0f}ms",
                                QPS=f"{qps:.2f}",
                            )
                            self.logger.info(
                                "[gen   ] batch %d | size=%d bucket=%s | "
                                "emb=%.1fms ret=%.1fms gen=%.1fms | "
                                "queries=%d/%d QPS=%.2f ms/tok=%.2f | elapsed=%.1fs",
                                seen_batches, len(payload.microbatch.queries),
                                payload.microbatch.bucket,
                                payload.embedding_sec * 1000,
                                payload.retrieval_sec * 1000,
                                generation_sec * 1000,
                                done_queries, len(self.queries),
                                qps, ms_per_tok, elapsed,
                            )
                        pbar.update(len(payload.microbatch.queries))
            except Exception as exc:  # pragma: no cover - runtime guard
                error_queue.put(exc)

        t_embed = threading.Thread(target=embed_worker, name="embed-worker", daemon=True)
        t_retrieval = threading.Thread(target=retrieval_worker, name="retrieval-worker", daemon=True)
        t_generation = threading.Thread(target=generation_worker, name="generation-worker", daemon=True)
        t_embed.start()
        t_retrieval.start()
        t_generation.start()

        # Prime & dispatch all batches: for both async_plain and async_bucket, the scheduler's
        # pending queue holds all remaining microbatches. We drain them all into q_er first,
        # then send None. Embed_worker processes each batch sequentially without self-dispatch.
        while self.scheduler.has_pending():
            dispatch = self.scheduler.next_dispatch(
                gpu_available=self.gpu_available,
                gpu_mem_gb=self._detect_gpu_free_memory_gb(),
                q_er_len=0,
                q_rg_len=0,
                long_ratio=bucket_long_ratio,
            )
            if dispatch is None:
                break
            action_key = (
                f"xE{int(dispatch.action.get('xE', 0))}_"
                f"xR{int(dispatch.action.get('xR', 0))}"
            )
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            q_er.put(dispatch)
            with stats_lock:
                q_er_size += 1
                q_er_max_seen = max(q_er_max_seen, q_er_size)

        # Sentinel: signals end of q_er stream.
        q_er.put(None)
        t_embed.join()

        t_retrieval.join()
        t_generation.join()
        pbar.close()

        if not error_queue.empty():
            first_error = error_queue.get()
            raise RuntimeError(f"Async pipeline worker failed: {first_error}") from first_error

        max_q_er = q_er_max_seen
        max_q_rg = q_rg_max_seen

        wall_time_sec = max(1e-9, time.perf_counter() - phase_start)
        execution_mode = "async_threaded_pipeline_plain" if pipeline_mode == "async_plain" else "async_threaded_pipeline_bucket"
        return self._build_summary(
            mode=scheduler_mode,
            execution=execution_mode,
            total_embedding_sec=total_embedding_sec,
            total_retrieval_sec=total_retrieval_sec,
            total_generation_sec=total_generation_sec,
            total_generated_tokens=total_generated_tokens,
            records=records,
            samples=samples,
            bucket_counts=bucket_counts,
            action_counts=action_counts,
            max_q_er=max_q_er,
            max_q_rg=max_q_rg,
            wall_time_sec=wall_time_sec,
            dispatch_trace=self.scheduler.dispatch_trace,
            feedback_trace=self.scheduler.feedback_trace,
            chunk_trace=chunk_trace,
            scheduler_dispatch_sec_total=scheduler_dispatch_sec_total,
            scheduler_feedback_sec_total=scheduler_feedback_sec_total,
            generation_target_trace=self.scheduler.generation_target_trace,
            device_plan_trace=self.scheduler.device_plan_trace,
            batch_shaping_trace=self.scheduler.batch_shaping_trace,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone RAG pipeline with manual controls: b (batch), xE (embedding device), "
            "xR (retrieval device), nprobe (retrieval search depth). "
            "No dependence on HedraRAG internals."
        )
    )

    parser.add_argument("--index-path", type=str, required=True, help="FAISS index path.")
    parser.add_argument("--corpus-path", type=str, required=True, help="Corpus path or HuggingFace dataset id.")
    parser.add_argument("--generator-model", type=str, required=True, help="Generator model path/id for vLLM.")

    parser.add_argument("--b", type=int, required=True, help="Manual query batch size.")
    parser.add_argument("--xE", type=int, choices=[0, 1], required=True, help="Embedding device: 0=CPU, 1=GPU.")
    parser.add_argument("--xR", type=int, choices=[0, 1], required=True, help="Retrieval device: 0=CPU, 1=GPU.")
    parser.add_argument("--nprobe", type=int, default=128, help="Retrieval search depth (default: 128).")
    parser.add_argument("--topk", type=int, default=1, help="Number of retrieved docs per query.")

    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-max-length", type=int, default=384)
    parser.add_argument("--pooling-method", type=str, default="mean", choices=["mean", "cls", "pooler"])
    parser.add_argument("--embedding-use-fp16", action="store_true")

    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Question: {query}\nContext: {context}\nAnswer:",
    )

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=None,
        help="GPU id to force via CUDA_VISIBLE_DEVICES. Leave unset to respect existing environment.",
    )

    parser.add_argument("--queries-file", type=str, default=None, help="Optional .txt/.jsonl/.json query file.")
    parser.add_argument("--query-field", type=str, default="question")
    parser.add_argument("--dataset-name", type=str, default="natural_questions")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--sample-queries", type=int, default=256)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--show-samples", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        choices=["serial", "async_plain", "async_bucket"],
        default="async_bucket",
        help="Execution mode for comparison experiments.",
    )

    parser.add_argument("--length-short-threshold", type=int, default=48)
    parser.add_argument("--length-long-threshold", type=int, default=128)
    parser.add_argument("--bucket-batch-short", type=int, default=64)
    parser.add_argument("--bucket-batch-mid", type=int, default=32)
    parser.add_argument("--bucket-batch-long", type=int, default=16)

    parser.add_argument("--embed-long-gpu-threshold", type=int, default=128)
    parser.add_argument("--retrieve-gpu-batch-threshold", type=int, default=64)

    # Memory-aware scheduling parameters (resource-constrained scenarios)
    parser.add_argument(
        "--enable-memory-aware-scheduling",
        action="store_const",
        const=True,
        default=True,
        help="Enable memory-aware action selection and batch shaping (default: True). "
             "Disable via --disable-memory-aware-scheduling.",
    )
    parser.add_argument(
        "--disable-memory-aware-scheduling",
        action="store_const",
        const=False,
        dest="enable_memory_aware_scheduling",
        help="Disable memory-aware scheduling.",
    )
    parser.add_argument(
        "--gpu-mem-low-threshold-gb",
        type=float,
        default=4.0,
        help="GPU free memory below this threshold triggers 'high' memory pressure (default: 4.0 GiB).",
    )
    parser.add_argument(
        "--gpu-mem-medium-threshold-gb",
        type=float,
        default=10.0,
        help="GPU free memory below this threshold triggers 'medium' pressure (default: 10.0 GiB).",
    )
    parser.add_argument(
        "--gpu-mem-high-batch-penalty",
        type=float,
        default=50.0,
        help="Score penalty applied to GPU-heavy actions (xR=1) under high memory pressure (default: 50.0).",
    )
    parser.add_argument(
        "--faiss-index-gb",
        type=float,
        default=2.0,
        help="Estimated FAISS index GPU memory footprint for scheduling decisions (default: 2.0 GiB).",
    )

    parser.add_argument("--backpressure-high", type=int, default=8)
    parser.add_argument("--scheduler-ema-alpha", type=float, default=0.25)
    parser.add_argument(
        "--scheduler-mode-choice",
        type=str,
        choices=["legacy_bucket", "generation_target_v1"],
        default="legacy_bucket",
    )
    parser.add_argument("--enable-batch-shaping", action="store_true")
    parser.add_argument("--ablate-bucketing", action="store_true")
    parser.add_argument("--ablate-online-batch", action="store_true")
    parser.add_argument("--ablate-online-action", action="store_true")
    parser.add_argument("--ablate-chunking", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu_id is not None and str(args.gpu_id).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    set_random_seed(args.seed)
    pipeline = StandaloneRAGPipeline(args)
    summary = pipeline.run()

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Saved summary to %s", output_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
