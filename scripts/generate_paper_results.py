#!/usr/bin/env python3
"""Generate LaTeX macros and tables from committed experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_NAMES = {
    "word-tfidf": "Word TF--IDF",
    "word-bm25": "BM25",
    "char-tfidf": "Char TF--IDF",
    "multilingual-e5-small": "mE5-small",
    "skillrouter-emb-0.6b": "SR-Emb-0.6B",
    "skillrouter+char-rrf": "SR+Char RRF",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def ms(value: float) -> str:
    return f"{value:.1f}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def significance_text(payload: dict, left: str, right: str, language: str) -> str:
    result = next(
        item
        for item in payload["significance"]
        if item["left"] == left
        and item["right"] == right
        and item["language"] == language
    )
    p_value = result["p_value_two_sided"]
    rendered = "$p<.001$" if p_value < 0.001 else f"$p={p_value:.3f}$"
    return (
        f"{rendered}, exact McNemar, "
        f"{result['right_only']} gains vs. {result['left_only']} losses"
    )


def write_macros(public: dict, output: Path) -> None:
    results = {item["name"]: item for item in public["results"]}
    values = [
        macro("PublicSkillCount", f"{public['skill_count']:,}"),
        macro("PublicTaskCount", str(public["task_count"])),
        macro("PublicVariantCount", str(public["query_variant_count"])),
        macro("PublicGoldCount", "182"),
    ]
    prefixes = {
        "word-tfidf": "Word",
        "word-bm25": "Bm",
        "char-tfidf": "Char",
        "multilingual-e5-small": "Efive",
        "skillrouter-emb-0.6b": "Sr",
        "skillrouter+char-rrf": "Fusion",
    }
    for method, prefix in prefixes.items():
        result = results[method]
        for language, lang_prefix in (
            ("en", "En"),
            ("zh", "Zh"),
            ("zh_instruction", "Inst"),
        ):
            metrics = result["by_language"][language]
            values.extend(
                [
                    macro(f"{prefix}{lang_prefix}Hit", pct(metrics["hit_at_1"])),
                    macro(f"{prefix}{lang_prefix}Recall", pct(metrics["recall_at_10"])),
                    macro(f"{prefix}{lang_prefix}Ndcg", f"{metrics['ndcg_at_10']:.3f}"),
                ]
            )
        values.append(macro(f"{prefix}Latency", ms(result["query_latency_ms"])))

    sparse_latencies = [
        results[name]["query_latency_ms"]
        for name in ("word-tfidf", "word-bm25", "char-tfidf")
    ]
    values.extend(
        [
            macro(
                "SparseLatencyRange",
                f"{min(sparse_latencies):.1f}--{max(sparse_latencies):.1f}",
            ),
            macro(
                "SrVsCharZhSignificance",
                significance_text(
                    public, "char-tfidf", "skillrouter-emb-0.6b", "zh"
                ),
            ),
            macro(
                "SrVsCharInstSignificance",
                significance_text(
                    public, "char-tfidf", "skillrouter-emb-0.6b", "zh_instruction"
                ),
            ),
        ]
    )
    output.write_text("\n".join(values) + "\n", encoding="utf-8")


def maybe_bold(value: float, best: float, formatted: str) -> str:
    return f"\\textbf{{{formatted}}}" if value == best else formatted


def write_public_table(public: dict, output: Path) -> None:
    metric_keys = [
        ("en", "hit_at_1"),
        ("zh", "hit_at_1"),
        ("zh_instruction", "hit_at_1"),
        ("en", "recall_at_10"),
        ("zh", "recall_at_10"),
        ("zh_instruction", "recall_at_10"),
    ]
    best = {
        key: max(item["by_language"][key[0]][key[1]] for item in public["results"])
        for key in metric_keys
    }
    rows = []
    for item in public["results"]:
        cells = []
        for key in metric_keys:
            value = item["by_language"][key[0]][key[1]]
            cells.append(maybe_bold(value, best[key], pct(value)))
        rows.append(
            f"{DISPLAY_NAMES[item['name']]} & "
            + " & ".join(cells)
            + f" & {item['query_latency_ms']:.1f} \\\\"
        )
    content = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{lrrrrrrr}
\toprule
& \multicolumn{3}{c}{Hit@1} & \multicolumn{3}{c}{Recall@10} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & EN & ZH & ZH+Inst & EN & ZH & ZH+Inst & ms/query \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Metadata-only retrieval on the SkillRouter hard tier.  All methods
rank the same 79,141 English skill names and descriptions.  Bold marks the best
value in each quality column.  Latency excludes one-time index construction.}
\label{tab:public}
\end{table*}
"""
    output.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public",
        type=Path,
        default=Path("results/skillrouter_multilingual.json"),
    )
    parser.add_argument(
        "--paper-dir", type=Path, default=Path("paper")
    )
    args = parser.parse_args()

    public = json.loads(args.public.read_text(encoding="utf-8"))
    write_macros(public, args.paper_dir / "generated-results.tex")
    write_public_table(public, args.paper_dir / "table-public-results.tex")
    print("generated LaTeX result macros and public table")


if __name__ == "__main__":
    main()
