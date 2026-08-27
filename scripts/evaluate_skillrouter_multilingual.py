#!/usr/bin/env python3
"""Evaluate multilingual skill retrieval on the public SkillRouter hard tier."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
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
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


LANGUAGES = ("en", "zh", "mixed")
PRIVACY_EXCLUDED_TASKS = {"court-form-filling"}
TOP_K = 20
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20270827


@dataclass
class RetrievalRun:
    name: str
    rankings: dict[str, dict[str, list[str]]]
    query_latency_ms: float
    index_seconds: float
    index_details: dict


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
                        "source": item.get("source", ""),
                    }
                )
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("skill pool contains duplicate skill_id values")
    return skill_ids, documents, metadata


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


def top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k >= scores.shape[0]:
        return np.argsort(-scores)
    selected = np.argpartition(-scores, top_k - 1)[:top_k]
    return selected[np.argsort(-scores[selected])]


def sparse_retrieval(
    name: str,
    vectorizer: TfidfVectorizer,
    documents: list[str],
    skill_ids: list[str],
    queries: list[dict],
) -> RetrievalRun:
    start = time.perf_counter()
    document_matrix = vectorizer.fit_transform(documents)
    index_seconds = time.perf_counter() - start

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
            indices = top_indices(scores, TOP_K)
            rankings[language][query["task_id"]] = [skill_ids[index] for index in indices]
        query_seconds += time.perf_counter() - start
        query_count += len(texts)

    return RetrievalRun(
        name=name,
        rankings=dict(rankings),
        query_latency_ms=1000 * query_seconds / query_count,
        index_seconds=index_seconds,
        index_details={
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
    index_seconds = time.perf_counter() - start

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
            indices = top_indices(scores, TOP_K)
            rankings[language][query["task_id"]] = [skill_ids[index] for index in indices]
        query_seconds += time.perf_counter() - start
        query_count += len(texts)

    return RetrievalRun(
        name="word-bm25",
        rankings=dict(rankings),
        query_latency_ms=1000 * query_seconds / query_count,
        index_seconds=index_seconds,
        index_details={
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
    documents: list[str],
    skill_ids: list[str],
    queries: list[dict],
    cache_path: Path,
    device: str,
    batch_size: int,
) -> RetrievalRun:
    resolved_device = detect_device(device)
    model = SentenceTransformer(model_name, device=resolved_device)
    model.max_seq_length = 512

    start = time.perf_counter()
    if cache_path.is_file():
        document_embeddings = np.load(cache_path, mmap_mode="r")
        index_source = "cache"
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        document_embeddings = model.encode(
            [f"passage: {document}" for document in documents],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(cache_path, document_embeddings)
        index_source = "encoded"
    index_seconds = time.perf_counter() - start

    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)
    query_count = 0
    query_seconds = 0.0
    document_matrix = np.asarray(document_embeddings)
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
            indices = top_indices(scores[row], TOP_K)
            rankings[language][query["task_id"]] = [skill_ids[index] for index in indices]
        query_seconds += time.perf_counter() - start
        query_count += len(texts)

    return RetrievalRun(
        name="multilingual-e5-small",
        rankings=dict(rankings),
        query_latency_ms=1000 * query_seconds / query_count,
        index_seconds=index_seconds,
        index_details={
            "model": model_name,
            "device": resolved_device,
            "max_seq_length": model.max_seq_length,
            "batch_size": batch_size,
            "embedding_shape": list(document_matrix.shape),
            "index_source": index_source,
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
        index_seconds=sum(run.index_seconds for run in runs),
        index_details={"members": [run.name for run in runs], "rrf_constant": 60},
    )


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def query_metrics(ranked: list[str], item: dict) -> dict[str, float]:
    core = set(item["core_gt_ids"])
    graded = item["relevance"]
    first_rank = next((index for index, skill_id in enumerate(ranked, start=1) if skill_id in core), None)
    retrieved_core = len(core.intersection(ranked[:10]))
    gains = [int(graded.get(skill_id, 0)) for skill_id in ranked[:10]]
    ideal = sorted((int(value) for value in graded.values()), reverse=True)[:10]
    ideal_dcg = dcg(ideal)
    return {
        "hit_at_1": float(first_rank == 1),
        "hit_at_5": float(first_rank is not None and first_rank <= 5),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
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


def aggregate_run(run: RetrievalRun, queries: list[dict], relevance: dict) -> dict:
    by_language = {}
    per_query = {}
    for language in LANGUAGES:
        rows = []
        for query in queries:
            task_id = query["task_id"]
            metrics = query_metrics(run.rankings[language][task_id], relevance[task_id])
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
        "index_seconds": round(run.index_seconds, 3),
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
    for language in ("zh", "mixed"):
        for baseline in ("word-tfidf", "word-bm25", "char-tfidf"):
            left = [
                indexed[baseline]["per_query"][f"{language}:{query['task_id']}"]["hit_at_1"]
                for query in queries
            ]
            right = [
                indexed["multilingual-e5-small"]["per_query"][
                    f"{language}:{query['task_id']}"
                ]["hit_at_1"]
                for query in queries
            ]
            comparisons.append(
                {
                    "language": language,
                    "left": baseline,
                    "right": "multilingual-e5-small",
                    **exact_mcnemar(left, right),
                }
            )
    return comparisons


def plot_results(results: list[dict], output: Path) -> None:
    methods = [result["name"] for result in results]
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    colors = {"en": "#4C78A8", "zh": "#F58518", "mixed": "#54A24B"}
    for offset, language in enumerate(LANGUAGES):
        values = [result["by_language"][language]["hit_at_1"] for result in results]
        ax.bar(x + (offset - 1) * width, values, width, label=language, color=colors[language])
    ax.set_ylabel("Hit@1")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.legend(frameon=False, ncol=3)
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
        "| Method | EN H@1 | ZH H@1 | MIX H@1 | EN R@10 | ZH R@10 | MIX R@10 | ms/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        lang = result["by_language"]
        lines.append(
            f"| {result['name']} | {lang['en']['hit_at_1']:.1%} | "
            f"{lang['zh']['hit_at_1']:.1%} | {lang['mixed']['hit_at_1']:.1%} | "
            f"{lang['en']['recall_at_10']:.1%} | {lang['zh']['recall_at_10']:.1%} | "
            f"{lang['mixed']['recall_at_10']:.1%} | {result['query_latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Hit@1 confidence intervals use 10,000 task-level bootstrap samples with seed "
            f"{BOOTSTRAP_SEED}. The privacy-filtered court-form task is excluded.",
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
        "--model", default="intfloat/multilingual-e5-small"
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("data/external/cache/multilingual-e5-small-hard-metadata.npy"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=Path("results/skillrouter_multilingual.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("results/skillrouter_multilingual.md")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("paper/figures/skillrouter-hit1.pdf")
    )
    args = parser.parse_args()

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
        args.model,
        documents,
        skill_ids,
        queries,
        args.embedding_cache,
        args.device,
        args.batch_size,
    )
    fusion = rrf_retrieval("e5+char-rrf", [dense, char], queries)
    runs = [word, bm25, char, dense, fusion]
    full_results = [aggregate_run(run, queries, relevance) for run in runs]

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
            "pid": os.getpid(),
        },
        "results": [compact_result(result) for result in full_results],
        "significance": significance(full_results, queries),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(payload, args.markdown)
    plot_results(full_results, args.figure)
    print(
        f"evaluated {len(queries)} tasks x {len(LANGUAGES)} languages "
        f"against {len(skill_ids)} skills"
    )


if __name__ == "__main__":
    main()
