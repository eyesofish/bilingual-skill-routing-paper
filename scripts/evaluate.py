#!/usr/bin/env python3
"""Evaluate deterministic bilingual skill-routing baselines."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


FIELD_WEIGHTS = {
    "name": 5.0,
    "aliases": 5.0,
    "tags": 4.0,
    "description": 3.0,
    "when_to_use": 3.0,
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", re.IGNORECASE)
CHAR_PATTERN = re.compile(r"[a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)


@dataclass(frozen=True)
class RankedSkill:
    name: str
    score: float


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize(text))


def char_ngrams(text: str, sizes: tuple[int, ...] = (2, 3, 4)) -> Counter[str]:
    compact = "".join(CHAR_PATTERN.findall(normalize(text)))
    grams: Counter[str] = Counter()
    for size in sizes:
        grams.update(compact[index : index + size] for index in range(len(compact) - size + 1))
    return grams


def fields(skill: dict) -> dict[str, str]:
    return {
        "name": skill["name"],
        "aliases": " ".join(skill.get("aliases", [])),
        "tags": " ".join(skill.get("tags", [])),
        "description": skill.get("description", ""),
        "when_to_use": skill.get("when_to_use", ""),
    }


def weighted_token_rank(catalog: list[dict], query: str) -> list[RankedSkill]:
    q = normalize(query)
    query_tokens = tokens(query)
    ranked: list[RankedSkill] = []

    for skill in catalog:
        score = 0.0
        skill_fields = fields(skill)
        if normalize(skill["name"]) == q:
            score += 100.0
        if any(normalize(alias) == q for alias in skill.get("aliases", [])):
            score += 90.0

        for field_name, text in skill_fields.items():
            normalized_text = normalize(text)
            weight = FIELD_WEIGHTS[field_name]
            if len(q) >= 2 and q in normalized_text:
                score += weight * 2.0
            if len(query_tokens) > 1:
                field_tokens = set(tokens(text))
                score += weight * sum(token in field_tokens for token in query_tokens)

        if score > 0:
            ranked.append(RankedSkill(skill["name"], score))

    return sorted(ranked, key=lambda item: (-item.score, item.name))


class CharNgramRanker:
    def __init__(self, catalog: list[dict]):
        self.catalog = catalog
        self.documents = {
            skill["name"]: char_ngrams(" ".join(fields(skill).values())) for skill in catalog
        }
        document_frequency: Counter[str] = Counter()
        for document in self.documents.values():
            document_frequency.update(document.keys())
        count = len(catalog)
        self.idf = {
            gram: math.log((count + 1) / (frequency + 1)) + 1.0
            for gram, frequency in document_frequency.items()
        }
        self.document_vectors = {
            name: self._tfidf(vector) for name, vector in self.documents.items()
        }
        self.document_norms = {
            name: math.sqrt(sum(value * value for value in vector.values()))
            for name, vector in self.document_vectors.items()
        }

    def _tfidf(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {
            gram: (frequency / total) * self.idf.get(gram, 1.0)
            for gram, frequency in counts.items()
        }

    def rank(self, query: str) -> list[RankedSkill]:
        query_vector = self._tfidf(char_ngrams(query))
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if query_norm == 0:
            return []

        ranked: list[RankedSkill] = []
        for skill in self.catalog:
            name = skill["name"]
            document_vector = self.document_vectors[name]
            dot = sum(
                value * document_vector.get(gram, 0.0) for gram, value in query_vector.items()
            )
            denominator = query_norm * self.document_norms[name]
            score = dot / denominator if denominator else 0.0
            if score > 0:
                ranked.append(RankedSkill(name, score))
        return sorted(ranked, key=lambda item: (-item.score, item.name))


def rrf_rank(
    token_results: list[RankedSkill],
    char_results: list[RankedSkill],
    constant: int = 60,
) -> list[RankedSkill]:
    scores: Counter[str] = Counter()
    for results in (token_results, char_results):
        for rank, item in enumerate(results, start=1):
            scores[item.name] += 1.0 / (constant + rank)
    return [
        RankedSkill(name, score)
        for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def apply_augments(catalog: list[dict], augments: dict[str, list[str]]) -> list[dict]:
    augmented: list[dict] = []
    for skill in catalog:
        updated = dict(skill)
        aliases = list(skill.get("aliases", []))
        for alias in augments.get(skill["name"], []):
            if alias not in aliases:
                aliases.append(alias)
        updated["aliases"] = aliases
        augmented.append(updated)
    return augmented


def choose_threshold(
    queries: list[dict],
    rankings: dict[str, list[RankedSkill]],
) -> tuple[float, float]:
    top_scores = [results[0].score for results in rankings.values() if results]
    candidates = {0.0, (max(top_scores) + 1e-9) if top_scores else 1.0}
    candidates.update(top_scores)

    best_threshold = 0.0
    best_accuracy = -1.0
    best_abstention = -1.0
    negatives = [query for query in queries if query["gold"] is None]
    for threshold in sorted(candidates):
        correct = 0
        negative_correct = 0
        for query in queries:
            results = rankings[query["id"]]
            predicted = results[0].name if results and results[0].score >= threshold else None
            is_correct = predicted == query["gold"]
            correct += int(is_correct)
            if query["gold"] is None:
                negative_correct += int(is_correct)
        accuracy = correct / len(queries)
        abstention = negative_correct / len(negatives) if negatives else 1.0
        key = (accuracy, abstention, -threshold)
        best_key = (best_accuracy, best_abstention, -best_threshold)
        if key > best_key:
            best_threshold = threshold
            best_accuracy = accuracy
            best_abstention = abstention
    return best_threshold, best_accuracy


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def measure_latency(
    ranker: Callable[[str], list[RankedSkill]],
    queries: list[dict],
    repeats: int,
) -> dict[str, float]:
    for query in queries:
        ranker(query["query"])
    samples_ms: list[float] = []
    for query in queries:
        start = time.perf_counter_ns()
        for _ in range(repeats):
            ranker(query["query"])
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000 / repeats
        samples_ms.append(elapsed_ms)
    return {
        "median_ms": round(statistics.median(samples_ms), 4),
        "p95_ms": round(percentile(samples_ms, 0.95), 4),
        "max_ms": round(max(samples_ms), 4),
    }


def evaluate_configuration(
    name: str,
    ranker: Callable[[str], list[RankedSkill]],
    queries: list[dict],
    repeats: int,
) -> dict:
    rankings = {query["id"]: ranker(query["query"]) for query in queries}
    dev = [query for query in queries if query["split"] == "dev"]
    test = [query for query in queries if query["split"] == "test"]
    threshold, dev_accuracy = choose_threshold(dev, rankings)

    positives = [query for query in test if query["gold"] is not None]
    negatives = [query for query in test if query["gold"] is None]
    reciprocal_ranks: list[float] = []
    hit1 = hit3 = routed_correct = negative_correct = 0
    by_language: dict[str, list[int]] = {}
    errors: list[dict] = []

    for query in test:
        results = rankings[query["id"]]
        predicted = results[0].name if results and results[0].score >= threshold else None
        routed_correct += int(predicted == query["gold"])
        if query["gold"] is None:
            negative_correct += int(predicted is None)
            if predicted is not None:
                errors.append(
                    {
                        "id": query["id"],
                        "gold": None,
                        "predicted": predicted,
                        "score": round(results[0].score, 6),
                    }
                )
            continue

        names = [item.name for item in results]
        rank = names.index(query["gold"]) + 1 if query["gold"] in names else None
        current_hit1 = int(rank == 1)
        hit1 += current_hit1
        hit3 += int(rank is not None and rank <= 3)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        by_language.setdefault(query["language"], []).append(current_hit1)
        if predicted != query["gold"]:
            errors.append(
                {
                    "id": query["id"],
                    "gold": query["gold"],
                    "predicted": predicted,
                    "raw_top": results[0].name if results else None,
                    "score": round(results[0].score, 6) if results else 0.0,
                }
            )

    return {
        "name": name,
        "threshold_from_dev": round(threshold, 8),
        "dev_routing_accuracy": round(dev_accuracy, 4),
        "test": {
            "queries": len(test),
            "positive_queries": len(positives),
            "negative_queries": len(negatives),
            "hit_at_1": round(hit1 / len(positives), 4),
            "hit_at_3": round(hit3 / len(positives), 4),
            "mrr": round(statistics.mean(reciprocal_ranks), 4),
            "abstention_accuracy": round(
                negative_correct / len(negatives) if negatives else 1.0, 4
            ),
            "routing_accuracy": round(routed_correct / len(test), 4),
            "hit_at_1_by_language": {
                language: round(sum(values) / len(values), 4)
                for language, values in sorted(by_language.items())
            },
        },
        "latency": measure_latency(ranker, queries, repeats),
        "errors": errors,
    }


def validate_queries(catalog: list[dict], queries: Iterable[dict]) -> None:
    skill_names = {skill["name"] for skill in catalog}
    seen_ids: set[str] = set()
    for query in queries:
        if query["id"] in seen_ids:
            raise ValueError(f"duplicate query id: {query['id']}")
        seen_ids.add(query["id"])
        if query["gold"] is not None and query["gold"] not in skill_names:
            raise ValueError(f"query {query['id']} references missing skill {query['gold']}")
        if query["split"] not in {"dev", "test"}:
            raise ValueError(f"invalid split for {query['id']}: {query['split']}")
        if query["language"] not in {"en", "zh", "mixed"}:
            raise ValueError(f"invalid language for {query['id']}: {query['language']}")


def write_markdown(result: dict, path: Path) -> None:
    rows = []
    for run in result["runs"]:
        test = run["test"]
        latency = run["latency"]
        rows.append(
            "| {name} | {hit1:.1%} | {hit3:.1%} | {mrr:.3f} | {abstain:.1%} | "
            "{routing:.1%} | {median:.4f} | {p95:.4f} |".format(
                name=run["name"],
                hit1=test["hit_at_1"],
                hit3=test["hit_at_3"],
                mrr=test["mrr"],
                abstain=test["abstention_accuracy"],
                routing=test["routing_accuracy"],
                median=latency["median_ms"],
                p95=latency["p95_ms"],
            )
        )

    content = [
        "# Baseline Results",
        "",
        f"- Catalog: {result['catalog_size']} skills",
        f"- Queries: {result['query_count']} total "
        f"({result['dev_query_count']} dev, {result['test_query_count']} test)",
        f"- Latency repeats per query: {result['latency_repeats']}",
        f"- Generated: {result['generated_at']}",
        "",
        "| Configuration | Hit@1 | Hit@3 | MRR | Abstention | Routing accuracy | "
        "Median ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Thresholds are selected only on the development split. Hit@k and MRR use",
        "positive test queries. Abstention accuracy uses negative test queries.",
        "Routing accuracy combines positive selection and negative abstention.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.json"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries.json"))
    parser.add_argument("--augments", type=Path, default=Path("data/metadata_augments.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/baseline.md"))
    parser.add_argument("--latency-repeats", type=int, default=100)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))["skills"]
    queries = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
    augments = json.loads(args.augments.read_text(encoding="utf-8"))
    validate_queries(catalog, queries)

    runs: list[dict] = []
    for variant_name, variant_catalog in (
        ("original", catalog),
        ("bilingual-augmented", apply_augments(catalog, augments)),
    ):
        char_ranker = CharNgramRanker(variant_catalog)
        token_ranker = lambda query, current=variant_catalog: weighted_token_rank(current, query)
        char_callable = char_ranker.rank
        fusion_ranker = lambda query, token=token_ranker, char=char_callable: rrf_rank(
            token(query), char(query)
        )
        for method_name, ranker in (
            ("weighted-token", token_ranker),
            ("char-ngram", char_callable),
            ("rrf-fusion", fusion_ranker),
        ):
            runs.append(
                evaluate_configuration(
                    f"{variant_name}/{method_name}",
                    ranker,
                    queries,
                    args.latency_repeats,
                )
            )

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "catalog_size": len(catalog),
        "query_count": len(queries),
        "dev_query_count": sum(query["split"] == "dev" for query in queries),
        "test_query_count": sum(query["split"] == "test" for query in queries),
        "latency_repeats": args.latency_repeats,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(result, args.markdown)
    print(f"wrote {len(runs)} runs to {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()

