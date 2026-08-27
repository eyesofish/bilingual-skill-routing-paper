from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.evaluate_skillrouter_multilingual import (
    dcg,
    exact_mcnemar,
    file_sha256,
    ordered_text_sha256,
    query_metrics,
    top_indices,
    verify_embedding_cache,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_top_indices_are_score_sorted(self) -> None:
        scores = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float32)
        self.assertEqual(top_indices(scores, 3).tolist(), [1, 3, 2])

    def test_query_metrics_support_multiple_gold_skills(self) -> None:
        relevance = {
            "core_gt_ids": ["gold-a", "gold-b"],
            "relevance": {"gold-a": 3, "gold-b": 3, "degraded": 1},
        }
        metrics = query_metrics(["wrong", "gold-b", "degraded", "gold-a"], relevance)
        self.assertEqual(metrics["hit_at_1"], 0)
        self.assertEqual(metrics["hit_at_5"], 1)
        self.assertEqual(metrics["mrr_at_20"], 0.5)
        self.assertEqual(metrics["recall_at_10"], 1.0)
        self.assertGreater(metrics["ndcg_at_10"], 0)

    def test_ndcg_ideal_excludes_unavailable_candidates(self) -> None:
        relevance = {
            "core_gt_ids": ["gold"],
            "relevance": {"gold": 3, "unavailable-degraded": 3},
        }
        metrics = query_metrics(["gold"], relevance, {"gold"})
        self.assertEqual(metrics["ndcg_at_10"], 1.0)

    def test_dcg_rewards_earlier_relevant_items(self) -> None:
        self.assertGreater(dcg([3, 1, 0]), dcg([1, 3, 0]))

    def test_top_indices_use_stable_skill_id_ties(self) -> None:
        scores = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        skill_ids = ["skill-c", "skill-a", "skill-b"]
        skill_id_order = np.argsort(np.asarray(skill_ids))
        self.assertEqual(
            top_indices(scores, 2, skill_ids, skill_id_order).tolist(),
            [1, 2],
        )

    def test_mcnemar_counts_discordant_pairs(self) -> None:
        result = exact_mcnemar([1, 1, 0, 0], [1, 0, 1, 1])
        self.assertEqual(result["left_only"], 1)
        self.assertEqual(result["right_only"], 2)
        self.assertEqual(result["discordant"], 3)
        self.assertGreaterEqual(result["p_value_two_sided"], 0)
        self.assertLessEqual(result["p_value_two_sided"], 1)

    def test_ordered_hash_detects_candidate_reordering(self) -> None:
        self.assertNotEqual(
            ordered_text_sha256(["a", "b"]),
            ordered_text_sha256(["b", "a"]),
        )

    def test_embedding_cache_metadata_binds_model_and_corpus(self) -> None:
        embeddings = np.zeros((2, 3), dtype=np.float32)
        skill_ids = ["skill-a", "skill-b"]
        documents = ["alpha", "beta"]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.npy"
            np.save(cache_path, embeddings)
            metadata = {
                "model": "model-a",
                "model_revision": "rev-1",
                "skill_count": 2,
                "shape": [2, 3],
                "skill_id_sha256": ordered_text_sha256(skill_ids),
                "document_sha256": ordered_text_sha256(documents),
                "document_fields": "name. description",
                "sha256": file_sha256(cache_path),
                "max_seq_length": 512,
            }
            verify_embedding_cache(
                cache_path,
                embeddings,
                metadata,
                expected_model="model-a",
                expected_revision="rev-1",
                skill_ids=skill_ids,
                documents=documents,
                expected_fields="name. description",
                expected_truncation_key="max_seq_length",
                expected_truncation_value=512,
            )
            with self.assertRaises(ValueError):
                verify_embedding_cache(
                    cache_path,
                    embeddings,
                    metadata,
                    expected_model="model-a",
                    expected_revision="rev-1",
                    skill_ids=list(reversed(skill_ids)),
                    documents=documents,
                    expected_fields="name. description",
                    expected_truncation_key="max_seq_length",
                    expected_truncation_value=512,
                )
            with self.assertRaises(ValueError):
                verify_embedding_cache(
                    cache_path,
                    embeddings,
                    metadata,
                    expected_model="model-a",
                    expected_revision="rev-1",
                    skill_ids=skill_ids,
                    documents=documents,
                    expected_fields="name. description",
                    expected_truncation_key="max_seq_length",
                    expected_truncation_value=256,
                )


if __name__ == "__main__":
    unittest.main()
