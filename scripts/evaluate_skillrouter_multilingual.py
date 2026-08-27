#!/usr/bin/env python3
"""Evaluate multilingual skill retrieval on the public SkillRouter hard tier."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn
import torch
import torch.nn.functional as torch_functional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

LANGUAGES = ("en", "zh", "zh_instruction")
PRIVACY_EXCLUDED_TASKS = {"court-form-filling"}
TOP_K = 20
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20270827
SKILLROUTER_QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)


@dataclass
class RetrievalRun:
    name: str
    rankings: dict[str, dict[str, list[str]]]
    query_latency_ms: float
    setup_seconds: float
    index_details: dict


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_model_specs(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pool(shard_dir: Path) -> tuple[list[str], list[str], list[dict]]:
    skill_ids: list[str] = []
    documents: list[str] = []
    metadata: list[dict] = []
    for path in sorted(shard_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                skill_ids.append(item["skill_id"])
                documents.append(f"{item['name']}. {item.get('description') or ''}".strip())
                metadata.append(
                    {
                        "skill_id": item["skill_id"],
                        "name": item["name"],
                        "description": item.get("description") or "",
                        "source": item.get("source", ""),
                    }
                )
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("skill pool contains duplicate skill_id values")
    return skill_ids, documents, metadata


def ordered_text_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_revision(model) -> str:
    if hasattr(model, "_first_module"):
        first = model._first_module()
        config = getattr(getattr(first, "auto_model", None), "config", None)
    else:
        config = getattr(model, "config", None)
    return str(getattr(config, "_commit_hash", "") or "")


def verify_embedding_cache(
    cache_path: Path,
    embeddings: np.ndarray,
    metadata: dict,
    *,
    expected_model: str,
    expected_revision: str,
    skill_ids: list[str],
    documents: list[str],
    expected_fields: str,
    expected_truncation_key: str,
    expected_truncation_value: int,
) -> None:
    expected = {
        "model": expected_model,
        "model_revision": expected_revision,
        "skill_count": len(skill_ids),
        "shape": list(embeddings.shape),
        "skill_id_sha256": ordered_text_sha256(skill_ids),
        "document_sha256": ordered_text_sha256(documents),
        "document_fields": expected_fields,
        "sha256": file_sha256(cache_path),
        expected_truncation_key: expected_truncation_value,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"embedding cache metadata mismatch for {cache_path}: {mismatches}")


def load_queries(path: Path, relevance_path: Path) -> tuple[list[dict], dict]:
    queries = load_jsonl(path)
    relevance = json.loads(relevance_path.read_text(encoding="utf-8"))
    filtered = []
    for query in queries:
        task_id = query["task_id"]
        if task_id in PRIVACY_EXCLUDED_TASKS:
            continue
        if relevance[task_id]["task_type"] == "generic_only":
            continue
        filtered.append(query)
    return filtered, relevance


def query_texts(queries: list[dict], language: str) -> list[str]:
    return [query[f"query_{language}"] for query in queries]


def top_indices(
    scores: np.ndarray,
    top_k: int,
    skill_ids: list[str] | None = None,
    skill_id_order: np.ndarray | None = None,
) -> np.ndarray:
    if top_k >= scores.shape[0]:
        selected = np.arange(scores.shape[0])
    else:
        cutoff = np.partition(scores, -top_k)[-top_k]
        selected = np.flatnonzero(scores > cutoff)
        remaining = top_k - len(selected)
        if remaining > 0:
            if skill_ids is None or skill_id_order is None:
                ties = np.flatnonzero(scores == cutoff)[:remaining]
            else:
                ties = skill_id_order[scores[skill_id_order] == cutoff][:remaining]
            selected = np.concatenate([selected, ties])
    if skill_ids is None:
        return selected[np.argsort(-scores[selected], kind="stable")]
    skill_array = np.asarray(skill_ids)
    order = np.lexsort((skill_array[selected], -scores[selected]))
    return selected[order]


def sparse_retrieval(
    name: str,
    vectorizer: TfidfVectorizer,
    documents: list[str],
    skill_ids: list[str],
    queries: list[dict],
) -> RetrievalRun:
    start = time.perf_counter()
    document_matrix = vectorizer.fit_transform(documents)
    setup_seconds = time.perf_counter() - start
    skill_id_order = np.argsort(np.asarray(skill_ids))

    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)
    query_count = 0
    query_seconds = 0.0
    for language in LANGUAGES:
        texts = query_texts(queries, language)
        start = time.perf_counter()
        query_matrix = vectorizer.transform(texts)
        similarities = query_matrix @ document_matrix.T
        for row, query in enumerate(queries):
            scores = similarities.getrow(row).toarray().ravel()
            indices = top_indices(scores, TOP_K, skill_ids, skill_id_order)
            rankings[language][query["task_id"]] = [skill_ids[index] for index in indices]
        query_seconds += time.perf_counter() - start
        query_count += len(texts)

    return RetrievalRun(
        name=name,
        rankings=dict(rankings),
        query_latency_ms=1000 * query_seconds / query_count,
        setup_seconds=setup_seconds,
        index_details={
            "setup_type": "fit_sparse_index",
            "features": len(vectorizer.vocabulary_),
            "document_matrix_shape": list(document_matrix.shape),
            "document_matrix_nnz": int(document_matrix.nnz),
        },
    )


def bm25_retrieval(
    documents: list[str],
    skill_ids: list[str],
    queries: list[dict],
    k1: float = 1.2,
    b: float = 0.75,
) -> RetrievalRun:
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1), max_features=300_000)
    start = time.perf_counter()
    counts = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
    document_lengths = np.asarray(counts.sum(axis=1)).ravel()
    average_length = float(document_lengths.mean())
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((counts.shape[0] - document_frequency + 0.5) / (document_frequency + 0.5) + 1)

    row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    term_frequency = counts.data
    normalizer = k1 * (1 - b + b * document_lengths[row_ids] / average_length)
    counts.data = (
        idf[counts.indices]
        * term_frequency
        * (k1 + 1)
        / (term_frequency + normalizer)
    ).astype(np.float32)
    setup_seconds = time.perf_counter() - start
    skill_id_order = np.argsort(np.asarray(skill_ids))

    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)
    query_count = 0
    query_seconds = 0.0
    for language in LANGUAGES:
        texts = query_texts(queries, language)
        start = time.perf_counter()
        query_matrix = vectorizer.transform(texts).tocsr().astype(np.float32)
        query_matrix.data.fill(1.0)
        similarities = query_matrix @ counts.T
        for row, query in enumerate(queries):
            scores = similarities.getrow(row).toarray().ravel()
            indices = top_indices(scores, TOP_K, skill_ids, skill_id_order)
            rankings[language][query["task_id"]] = [skill_ids[index] for index in indices]
        query_seconds += time.perf_counter() - start
        query_count += len(texts)

    return RetrievalRun(
        name="word-bm25",
        rankings=dict(rankings),
        query_latency_ms=1000 * query_seconds / query_count,
        setup_seconds=setup_seconds,
        index_details={
            "setup_type": "fit_sparse_index",
            "features": len(vectorizer.vocabulary_),
            "document_matrix_shape": list(counts.shape),
            "document_matrix_nnz": int(counts.nnz),
            "k1": k1,
            "b": b,
        },
    )


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dense_retrieval(
    model_name: str,
    revision: str,
    documents: list[str],
    skill_ids: list[str],
    queries: list[dict],
    cache_path: Path,
    device: str,
    batch_size: int,
) -> RetrievalRun:
    resolved_device = detect_device(device)
    model = SentenceTransformer(
        model_name, revision=revision, device=resolved_device
    )
    model.max_seq_length = 512
    revision = model_revision(model)
    formatted_documents = [f"passage: {document}" for document in documents]

    start = time.perf_counter()
    if cache_path.is_file():
        document_embeddings = np.load(cache_path, mmap_mode="r")
        metadata = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
        verify_embedding_cache(
            cache_path,
            document_embeddings,
            metadata,
            expected_model=model_name,
            expected_revision=revision,
            skill_ids=skill_ids,
            documents=formatted_documents,
            expected_fields="passage: name. description",
            expected_truncation_key="max_seq_length",
            expected_truncation_value=model.max_seq_length,
        )
        index_source = "cache"
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        document_embeddings = model.encode(
            formatted_documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(cache_path, document_embeddings)
        index_source = "encoded"
    setup_seconds = time.perf_counter() - start

    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)
    query_count = 0
    query_seconds = 0.0
    document_matrix = np.asarray(document_embeddings)
    skill_id_order = np.argsort(np.asarray(skill_ids))
    for language in LANGUAGES:
        texts = query_texts(queries, language)
        start = time.perf_counter()
        query_embeddings = model.encode(
            [f"query: {text}" for text in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores = query_embeddings @ document_matrix.T
        for row, query in enumerate(queries):
            indices = top_indices(scores[row], TOP_K, skill_ids, skill_id_order)
            rankings[language][query["task_id"]] = [skill_ids[index] for index in indices]
        query_seconds += time.perf_counter() - start
        query_count += len(texts)

    return RetrievalRun(
        name="multilingual-e5-small",
        rankings=dict(rankings),
        query_latency_ms=1000 * query_seconds / query_count,
        setup_seconds=setup_seconds,
        index_details={
            "setup_type": "load_and_validate_cache",
            "model": model_name,
            "model_revision": revision,
            "device": resolved_device,
            "document_max_length": metadata["max_seq_length"],
            "query_max_length": model.max_seq_length,
            "batch_size": batch_size,
            "embedding_shape": list(document_matrix.shape),
            "index_source": index_source,
        },
    )


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def encode_skillrouter_texts(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            embeddings = last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            embeddings = torch_functional.normalize(embeddings, p=2, dim=1)
        batches.append(embeddings.float().cpu().numpy())
    return np.concatenate(batches).astype(np.float32)


def skillrouter_dense_retrieval(
    model_name: str,
    revision: str,
    skill_ids: list[str],
    documents: list[str],
    queries: list[dict],
    cache_path: Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> RetrievalRun:
    resolved_device = detect_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = model.eval().to(resolved_device)
    revision = model_revision(model)

    start = time.perf_counter()
    document_embeddings = np.load(cache_path, mmap_mode="r")
    metadata = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    verify_embedding_cache(
        cache_path,
        document_embeddings,
        metadata,
        expected_model=model_name,
        expected_revision=revision,
        skill_ids=skill_ids,
        documents=documents,
        expected_fields="name | description",
        expected_truncation_key="max_length",
        expected_truncation_value=max_length,
    )
    setup_seconds = time.perf_counter() - start

    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)
    query_count = 0
    query_seconds = 0.0
    document_matrix = np.asarray(document_embeddings)
    skill_id_order = np.argsort(np.asarray(skill_ids))
    for language in LANGUAGES:
        texts = [
            SKILLROUTER_QUERY_INSTRUCTION + text for text in query_texts(queries, language)
        ]
        start = time.perf_counter()
        query_embeddings = encode_skillrouter_texts(
            texts, tokenizer, model, batch_size, max_length
        )
        scores = query_embeddings @ document_matrix.T
        for row, query in enumerate(queries):
            indices = top_indices(scores[row], TOP_K, skill_ids, skill_id_order)
            rankings[language][query["task_id"]] = [skill_ids[index] for index in indices]
        query_seconds += time.perf_counter() - start
        query_count += len(texts)

    return RetrievalRun(
        name="skillrouter-emb-0.6b",
        rankings=dict(rankings),
        query_latency_ms=1000 * query_seconds / query_count,
        setup_seconds=setup_seconds,
        index_details={
            "setup_type": "load_and_validate_cache",
            "model": model_name,
            "model_revision": revision,
            "device": resolved_device,
            "document_max_length": metadata["max_length"],
            "query_max_length": max_length,
            "batch_size": batch_size,
            "embedding_shape": list(document_matrix.shape),
            "index_source": "cache",
            "document_fields": "name|description",
        },
    )


def rrf_retrieval(name: str, runs: list[RetrievalRun], queries: list[dict]) -> RetrievalRun:
    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)
    start = time.perf_counter()
    for language in LANGUAGES:
        for query in queries:
            task_id = query["task_id"]
            scores: dict[str, float] = defaultdict(float)
            for run in runs:
                for rank, skill_id in enumerate(run.rankings[language][task_id], start=1):
                    scores[skill_id] += 1.0 / (60 + rank)
            ordered = sorted(scores, key=lambda skill_id: (-scores[skill_id], skill_id))
            rankings[language][task_id] = ordered[:TOP_K]
    elapsed = time.perf_counter() - start
    return RetrievalRun(
        name=name,
        rankings=dict(rankings),
        query_latency_ms=sum(run.query_latency_ms for run in runs)
        + 1000 * elapsed / (len(queries) * len(LANGUAGES)),
        setup_seconds=sum(run.setup_seconds for run in runs),
        index_details={"members": [run.name for run in runs], "rrf_constant": 60},
    )


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def query_metrics(
    ranked: list[str], item: dict, available_skill_ids: set[str] | None = None
) -> dict[str, float]:
    core = set(item["core_gt_ids"])
    graded = item["relevance"]
    first_rank = next((index for index, skill_id in enumerate(ranked, start=1) if skill_id in core), None)
    retrieved_core = len(core.intersection(ranked[:10]))
    gains = [int(graded.get(skill_id, 0)) for skill_id in ranked[:10]]
    ideal = sorted(
        (
            int(value)
            for skill_id, value in graded.items()
            if available_skill_ids is None or skill_id in available_skill_ids
        ),
        reverse=True,
    )[:10]
    ideal_dcg = dcg(ideal)
    return {
        "hit_at_1": float(first_rank == 1),
        "hit_at_5": float(first_rank is not None and first_rank <= 5),
        "mrr_at_20": 1.0 / first_rank if first_rank else 0.0,
        "recall_at_10": retrieved_core / len(core),
        "ndcg_at_10": dcg(gains) / ideal_dcg if ideal_dcg else 0.0,
    }


def bootstrap_ci(values: list[float]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(float(np.mean(sample)))
    low, high = np.percentile(means, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def aggregate_run(
    run: RetrievalRun,
    queries: list[dict],
    relevance: dict,
    available_skill_ids: set[str],
) -> dict:
    by_language = {}
    per_query = {}
    for language in LANGUAGES:
        rows = []
        for query in queries:
            task_id = query["task_id"]
            metrics = query_metrics(
                run.rankings[language][task_id],
                relevance[task_id],
                available_skill_ids,
            )
            rows.append(metrics)
            per_query[f"{language}:{task_id}"] = metrics
        aggregated = {
            metric: round(float(np.mean([row[metric] for row in rows])), 4)
            for metric in rows[0]
        }
        aggregated["hit_at_1_ci95"] = bootstrap_ci([row["hit_at_1"] for row in rows])
        by_language[language] = aggregated
    return {
        "name": run.name,
        "by_language": by_language,
        "query_latency_ms": round(run.query_latency_ms, 3),
        "setup_seconds": round(run.setup_seconds, 3),
        "index_details": run.index_details,
        "per_query": per_query,
    }


def exact_mcnemar(left: list[float], right: list[float]) -> dict:
    left_only = sum(a == 1 and b == 0 for a, b in zip(left, right))
    right_only = sum(a == 0 and b == 1 for a, b in zip(left, right))
    discordant = left_only + right_only
    p_value = (
        float(scipy.stats.binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "p_value_two_sided": round(p_value, 6),
    }


def significance(results: list[dict], queries: list[dict]) -> list[dict]:
    indexed = {result["name"]: result for result in results}
    comparisons = []
    for language in ("zh", "zh_instruction"):
        for right_name in ("multilingual-e5-small", "skillrouter-emb-0.6b"):
            for baseline in ("word-tfidf", "word-bm25", "char-tfidf"):
                left = [
                    indexed[baseline]["per_query"][f"{language}:{query['task_id']}"][
                        "hit_at_1"
                    ]
                    for query in queries
                ]
                right = [
                    indexed[right_name]["per_query"][f"{language}:{query['task_id']}"][
                        "hit_at_1"
                    ]
                    for query in queries
                ]
                comparisons.append(
                    {
                        "language": language,
                        "left": baseline,
                        "right": right_name,
                        **exact_mcnemar(left, right),
                    }
                )
    return comparisons


def plot_results(results: list[dict], output: Path) -> None:
    methods = [result["name"] for result in results]
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    colors = {"en": "#4C78A8", "zh": "#F58518", "zh_instruction": "#54A24B"}
    for offset, language in enumerate(LANGUAGES):
        values = [result["by_language"][language]["hit_at_1"] for result in results]
        ax.bar(x + (offset - 1) * width, values, width, label=language, color=colors[language])
    ax.set_ylabel("Hit@1")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["EN", "ZH", "ZH+Inst"], frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def compact_result(result: dict) -> dict:
    clean = dict(result)
    clean.pop("per_query", None)
    return clean


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# SkillRouter Multilingual Results",
        "",
        f"- Candidate skills: {payload['skill_count']}",
        f"- Tasks: {payload['task_count']}",
        f"- Query variants: {payload['query_variant_count']}",
        f"- Tier: {payload['tier']}",
        "",
        "| Method | EN H@1 | ZH H@1 | ZH+Inst H@1 | EN R@10 | ZH R@10 | ZH+Inst R@10 | ms/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        lang = result["by_language"]
        lines.append(
            f"| {result['name']} | {lang['en']['hit_at_1']:.1%} | "
                f"{lang['zh']['hit_at_1']:.1%} | {lang['zh_instruction']['hit_at_1']:.1%} | "
            f"{lang['en']['recall_at_10']:.1%} | {lang['zh']['recall_at_10']:.1%} | "
                f"{lang['zh_instruction']['recall_at_10']:.1%} | {result['query_latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Hit@1 confidence intervals use 10,000 task-level bootstrap samples with seed "
            f"{BOOTSTRAP_SEED}. Source-excluded and generic-only tasks are omitted.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", type=Path, default=Path("data/external/skillrouter/hard")
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("data/skillrouter_queries_multilingual.jsonl")
    )
    parser.add_argument(
        "--relevance",
        type=Path,
        default=Path("data/external/skillrouter/relevance.json"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model_revisions.json"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("data/external/cache/multilingual-e5-small-hard-metadata.npy"),
    )
    parser.add_argument(
        "--skillrouter-cache",
        type=Path,
        default=Path("data/external/cache/skillrouter-0.6b-hard-metadata.npy"),
    )
    parser.add_argument("--skillrouter-batch-size", type=int, default=8)
    parser.add_argument("--skillrouter-max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=Path("results/skillrouter_multilingual.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("results/skillrouter_multilingual.md")
    )
    parser.add_argument(
        "--rankings-output",
        type=Path,
        default=Path("results/skillrouter_rankings.json"),
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("paper/figures/skillrouter-hit1.pdf")
    )
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=Path("config/experiment_environment.json"),
    )
    args = parser.parse_args()
    model_specs = load_model_specs(args.model_config)
    e5_spec = model_specs["multilingual_e5_small"]
    skillrouter_spec = model_specs["skillrouter_embedding_0_6b"]

    skill_ids, documents, metadata = load_pool(args.shards)
    queries, relevance = load_queries(args.queries, args.relevance)
    available = set(skill_ids)
    missing = {
        skill_id
        for query in queries
        for skill_id in relevance[query["task_id"]]["core_gt_ids"]
        if skill_id not in available
    }
    if missing:
        raise ValueError(f"missing {len(missing)} gold skill ids from the candidate pool")

    word = sparse_retrieval(
        "word-tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=300_000,
            dtype=np.float32,
        ),
        documents,
        skill_ids,
        queries,
    )
    bm25 = bm25_retrieval(documents, skill_ids, queries)
    char = sparse_retrieval(
        "char-tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=300_000,
            dtype=np.float32,
        ),
        documents,
        skill_ids,
        queries,
    )
    dense = dense_retrieval(
        e5_spec["model_id"],
        e5_spec["revision"],
        documents,
        skill_ids,
        queries,
        args.embedding_cache,
        args.device,
        args.batch_size,
    )
    skillrouter_documents = [
        f"{item['name']} | {item['description']}".strip() for item in metadata
    ]
    skillrouter_dense = skillrouter_dense_retrieval(
        skillrouter_spec["model_id"],
        skillrouter_spec["revision"],
        skill_ids,
        skillrouter_documents,
        queries,
        args.skillrouter_cache,
        args.device,
        args.skillrouter_batch_size,
        args.skillrouter_max_length,
    )
    fusion = rrf_retrieval("skillrouter+char-rrf", [skillrouter_dense, char], queries)
    runs = [word, bm25, char, dense, skillrouter_dense, fusion]
    full_results = [
        aggregate_run(run, queries, relevance, available) for run in runs
    ]

    hardware = json.loads(args.environment_config.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier": "hard",
        "skill_count": len(skill_ids),
        "task_count": len(queries),
        "query_variant_count": len(queries) * len(LANGUAGES),
        "privacy_excluded_tasks": sorted(PRIVACY_EXCLUDED_TASKS),
        "top_k": TOP_K,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            **hardware,
        },
        "results": [compact_result(result) for result in full_results],
        "significance": significance(full_results, queries),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.rankings_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "top_k": TOP_K,
                "rankings": {run.name: run.rankings for run in runs},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.markdown)
    plot_results(full_results, args.figure)
    print(
        f"evaluated {len(queries)} tasks x {len(LANGUAGES)} languages "
        f"against {len(skill_ids)} skills"
    )


if __name__ == "__main__":
    main()
