#!/usr/bin/env python3
"""Validate the multilingual SkillRouter query extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PRIVACY_EXCLUDED_TASKS = {"court-form-filling"}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ZH_INSTRUCTION_PREFIX = (
    "请完成下面这个 task。Please follow every requirement，并原样保留所有 "
    "technical identifiers、file paths、API names、code symbols、"
    "numeric constraints 和 output schema：\n\n"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries", type=Path, default=Path("data/skillrouter_queries_multilingual.jsonl")
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/external/skillrouter/tasks.jsonl"),
    )
    parser.add_argument(
        "--relevance",
        type=Path,
        default=Path("data/external/skillrouter/relevance.json"),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/skillrouter_queries_multilingual.stats.json"),
    )
    args = parser.parse_args()

    source_tasks = {item["task_id"]: item for item in load_jsonl(args.tasks)}
    relevance = json.loads(args.relevance.read_text(encoding="utf-8"))
    expected_ids = {
        task_id
        for task_id, task in source_tasks.items()
        if not task["excluded"]
        and relevance[task_id]["task_type"] != "generic_only"
        and task_id not in PRIVACY_EXCLUDED_TASKS
    }
    queries = load_jsonl(args.queries)
    ids = [query["task_id"] for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task_id in multilingual query file")
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))
        extra = sorted(set(ids) - expected_ids)
        raise ValueError(f"query task mismatch: missing={missing}, extra={extra}")

    length_ratios = []
    for query in queries:
        task_id = query["task_id"]
        if query["query_en"] != source_tasks[task_id]["instruction_text"]:
            raise ValueError(f"English source changed for {task_id}")
        if not CJK.search(query["query_zh"]):
            raise ValueError(f"Chinese query has no CJK text: {task_id}")
        if query["query_zh_instruction"] != ZH_INSTRUCTION_PREFIX + query["query_zh"]:
            raise ValueError(f"invalid fixed instruction wrapper: {task_id}")
        if not LATIN.search(query["query_zh_instruction"]):
            raise ValueError(f"instruction-prefixed query has no Latin text: {task_id}")
        combined = query["query_zh"] + "\n" + query["query_zh_instruction"]
        if EMAIL.search(combined):
            raise ValueError(f"possible personal identifier in translated query: {task_id}")
        length_ratios.append(len(query["query_zh"]) / max(1, len(query["query_en"])))

    stats = {
        "schema_version": 1,
        "task_count": len(queries),
        "query_variant_count": len(queries) * 3,
        "query_file_sha256": sha256(args.queries),
        "excluded_task_ids": sorted(PRIVACY_EXCLUDED_TASKS),
        "domains": dict(sorted(Counter(query["domain"] for query in queries).items())),
        "difficulties": dict(
            sorted(Counter(query["difficulty"] for query in queries).items())
        ),
        "mean_zh_to_en_character_ratio": round(sum(length_ratios) / len(length_ratios), 4),
        "min_zh_to_en_character_ratio": round(min(length_ratios), 4),
        "max_zh_to_en_character_ratio": round(max(length_ratios), 4),
    }
    args.stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"validated {stats['task_count']} tasks and "
        f"{stats['query_variant_count']} language variants"
    )


if __name__ == "__main__":
    main()
