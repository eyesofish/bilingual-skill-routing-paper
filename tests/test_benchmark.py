from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_catalog import build_catalog, parse_frontmatter
from scripts.evaluate import (
    CharNgramRanker,
    apply_augments,
    validate_queries,
    weighted_token_rank,
)


ROOT = Path(__file__).resolve().parents[1]


class FrontmatterTests(unittest.TestCase):
    def test_parses_folded_description_and_lists(self) -> None:
        metadata = parse_frontmatter(
            "\n".join(
                [
                    "---",
                    "name: demo",
                    "description: >",
                    "  First line.",
                    "  Second line.",
                    "aliases:",
                    "  - 示例",
                    "  - demo skill",
                    "tags: [one, two]",
                    "---",
                    "# Demo",
                ]
            )
        )
        self.assertEqual(metadata["name"], "demo")
        self.assertEqual(metadata["description"], "First line. Second line.")
        self.assertEqual(metadata["aliases"], ["示例", "demo skill"])
        self.assertEqual(metadata["tags"], ["one", "two"])


class CatalogTests(unittest.TestCase):
    def test_builds_sanitized_catalog(self) -> None:
        catalog = build_catalog(ROOT / "config/sources.json")
        self.assertGreaterEqual(len(catalog), 20)
        self.assertIn("code-review", {skill["name"] for skill in catalog})
        for skill in catalog:
            self.assertNotIn(str(Path.home()), skill["source_path"])

    def test_query_gold_labels_exist(self) -> None:
        catalog = build_catalog(ROOT / "config/sources.json")
        queries = json.loads((ROOT / "data/queries.json").read_text())["queries"]
        validate_queries(catalog, queries)


class RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = build_catalog(ROOT / "config/sources.json")
        cls.augments = json.loads((ROOT / "data/metadata_augments.json").read_text())

    def test_weighted_token_finds_existing_bilingual_alias(self) -> None:
        results = weighted_token_rank(self.catalog, "检查代码")
        self.assertEqual(results[0].name, "code-review")

    def test_augmentation_adds_cross_language_signal(self) -> None:
        augmented = apply_augments(self.catalog, self.augments)
        results = weighted_token_rank(augmented, "写论文")
        self.assertEqual(results[0].name, "research-paper-writing")

    def test_char_ngram_returns_ranked_results(self) -> None:
        ranker = CharNgramRanker(apply_augments(self.catalog, self.augments))
        results = ranker.rank("帮我做安全审查")
        self.assertTrue(results)
        self.assertEqual(results[0].name, "security-review")


if __name__ == "__main__":
    unittest.main()

