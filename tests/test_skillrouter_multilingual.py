from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_skillrouter_multilingual import (
    dcg,
    exact_mcnemar,
    query_metrics,
    top_indices,
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
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["recall_at_10"], 1.0)
        self.assertGreater(metrics["ndcg_at_10"], 0)

    def test_dcg_rewards_earlier_relevant_items(self) -> None:
        self.assertGreater(dcg([3, 1, 0]), dcg([1, 3, 0]))

    def test_mcnemar_counts_discordant_pairs(self) -> None:
        result = exact_mcnemar([1, 1, 0, 0], [1, 0, 1, 1])
        self.assertEqual(result["left_only"], 1)
        self.assertEqual(result["right_only"], 2)
        self.assertEqual(result["discordant"], 3)
        self.assertGreaterEqual(result["p_value_two_sided"], 0)
        self.assertLessEqual(result["p_value_two_sided"], 1)


if __name__ == "__main__":
    unittest.main()
