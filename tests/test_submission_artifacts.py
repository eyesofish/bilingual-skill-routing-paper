from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class SubmissionArtifactTests(unittest.TestCase):
    def test_multilingual_data_matches_recorded_checksum(self) -> None:
        query_path = ROOT / "data/skillrouter_queries_multilingual.jsonl"
        stats = json.loads(
            (ROOT / "data/skillrouter_queries_multilingual.stats.json").read_text()
        )
        self.assertEqual(stats["task_count"], 73)
        self.assertEqual(stats["query_variant_count"], 219)
        self.assertEqual(stats["query_file_sha256"], sha256(query_path))

    def test_public_result_is_complete_and_finite(self) -> None:
        result = json.loads(
            (ROOT / "results/skillrouter_multilingual.json").read_text()
        )
        self.assertEqual(result["skill_count"], 79141)
        self.assertEqual(result["task_count"], 73)
        self.assertEqual(result["query_variant_count"], 219)
        names = {item["name"] for item in result["results"]}
        self.assertEqual(
            names,
            {
                "word-tfidf",
                "word-bm25",
                "char-tfidf",
                "multilingual-e5-small",
                "skillrouter-emb-0.6b",
                "skillrouter+char-rrf",
            },
        )
        for item in result["results"]:
            self.assertEqual(
                set(item["by_language"]), {"en", "zh", "zh_instruction"}
            )
            for metrics in item["by_language"].values():
                for key in (
                    "hit_at_1",
                    "hit_at_5",
                    "mrr_at_20",
                    "recall_at_10",
                    "ndcg_at_10",
                ):
                    self.assertTrue(math.isfinite(metrics[key]))
                    self.assertGreaterEqual(metrics[key], 0)
                    self.assertLessEqual(metrics[key], 1)

    def test_model_revisions_are_exactly_pinned(self) -> None:
        models = json.loads((ROOT / "config/model_revisions.json").read_text())
        self.assertEqual(
            set(models), {"multilingual_e5_small", "skillrouter_embedding_0_6b"}
        )
        for item in models.values():
            self.assertRegex(item["revision"], r"^[0-9a-f]{40}$")

    def test_dataset_revision_is_exactly_pinned(self) -> None:
        datasets = json.loads(
            (ROOT / "config/dataset_revisions.json").read_text()
        )
        item = datasets["skillrouter_eval_core"]
        self.assertEqual(item["repo_id"], "pipizhao/SkillRouter-Eval-Core")
        self.assertRegex(item["revision"], r"^[0-9a-f]{40}$")

    def test_paper_has_no_placeholders(self) -> None:
        main = (ROOT / "paper/main.tex").read_text()
        self.assertNotIn("TODO", main)
        self.assertNotIn("TBD", main)
        self.assertNotIn("EfiveVsWord", main)
        checklist = (ROOT / "paper/reproducibility-checklist.tex").read_text()
        self.assertEqual(checklist.count("Type your response here"), 3)

    def test_final_pdfs_exist(self) -> None:
        paths = [
            ROOT / "paper/main.pdf",
            ROOT / "paper/supplement.pdf",
            ROOT / "paper/reproducibility-checklist.pdf",
        ]
        for path in paths:
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 10_000, path)


if __name__ == "__main__":
    unittest.main()
