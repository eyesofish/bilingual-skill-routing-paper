#!/usr/bin/env python3
"""Create task-level error and hubness analysis from saved rankings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import scipy.stats


LANGUAGES = ("en", "zh", "zh_instruction")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def hit_at_one(ranking: list[str], core: set[str]) -> bool:
    return bool(ranking and ranking[0] in core)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rankings", type=Path, default=Path("results/skillrouter_rankings.json")
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
        "--output", type=Path, default=Path("results/error_analysis.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("results/error_analysis.md")
    )
    args = parser.parse_args()

    rankings = json.loads(args.rankings.read_text(encoding="utf-8"))["rankings"]
    queries = load_jsonl(args.queries)
    relevance = json.loads(args.relevance.read_text(encoding="utf-8"))
    methods = list(rankings)
    analysis = {"methods": {}, "pairwise": {}}

    for method in methods:
        analysis["methods"][method] = {}
        for language in LANGUAGES:
            successes = []
            lengths = []
            top_predictions = []
            for query in queries:
                task_id = query["task_id"]
                core = set(relevance[task_id]["core_gt_ids"])
                ranking = rankings[method][language][task_id]
                successes.append(int(hit_at_one(ranking, core)))
                lengths.append(len(query[f"query_{language}"]))
                top_predictions.append(ranking[0])
            if len(set(successes)) == 1:
                correlation_statistic = 0.0
                correlation_p_value = 1.0
            else:
                correlation = scipy.stats.spearmanr(lengths, successes)
                correlation_statistic = float(correlation.statistic)
                correlation_p_value = float(correlation.pvalue)
            hubs = Counter(top_predictions).most_common(5)
            analysis["methods"][method][language] = {
                "hit_count": sum(successes),
                "task_count": len(successes),
                "unique_top1_predictions": len(set(top_predictions)),
                "top1_hubs": [{"skill_id": skill, "count": count} for skill, count in hubs],
                "query_length_hit_spearman": round(correlation_statistic, 4),
                "query_length_hit_p_value": round(correlation_p_value, 6),
            }

    for language in ("zh", "zh_instruction"):
        for left, right in (
            ("char-tfidf", "skillrouter-emb-0.6b"),
            ("skillrouter-emb-0.6b", "skillrouter+char-rrf"),
        ):
            gains = []
            losses = []
            shared = []
            for query in queries:
                task_id = query["task_id"]
                core = set(relevance[task_id]["core_gt_ids"])
                left_hit = hit_at_one(rankings[left][language][task_id], core)
                right_hit = hit_at_one(rankings[right][language][task_id], core)
                if right_hit and not left_hit:
                    gains.append(task_id)
                elif left_hit and not right_hit:
                    losses.append(task_id)
                elif left_hit and right_hit:
                    shared.append(task_id)
            analysis["pairwise"][f"{language}:{left}->{right}"] = {
                "gains": gains,
                "losses": losses,
                "shared_hits": shared,
            }

    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = ["# SkillRouter Error Analysis", ""]
    for method in methods:
        lines.append(f"## {method}")
        for language in LANGUAGES:
            item = analysis["methods"][method][language]
            hub = item["top1_hubs"][0]
            lines.append(
                f"- {language}: {item['hit_count']}/{item['task_count']} Hit@1; "
                f"{item['unique_top1_predictions']} unique top predictions; "
                f"largest hub `{hub['skill_id']}` appears {hub['count']} times; "
                f"length correlation rho={item['query_length_hit_spearman']:.3f} "
                f"(p={item['query_length_hit_p_value']:.3g})."
            )
        lines.append("")
    lines.append("## Pairwise task changes")
    for name, item in analysis["pairwise"].items():
        lines.append(
            f"- {name}: {len(item['gains'])} gains, {len(item['losses'])} losses, "
            f"{len(item['shared_hits'])} shared hits."
        )
        lines.append(f"  - gains: {', '.join(item['gains']) or 'none'}")
        lines.append(f"  - losses: {', '.join(item['losses']) or 'none'}")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
