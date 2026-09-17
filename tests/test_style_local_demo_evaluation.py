import unittest

from scripts.evaluate_style_local_demo_index import evaluate_rankings


class StyleLocalDemoEvaluationTests(unittest.TestCase):
    def test_role_aware_metrics_exclude_composite_from_direct_candidates(self) -> None:
        manifest = {
            "query_set_id": "test",
            "queries": [
                {
                    "query_id": "direct",
                    "query_text": "Deep House",
                    "primary_card_ids": ["direct-a"],
                    "expected_card_ids": ["direct-a"],
                },
                {
                    "query_id": "composite",
                    "query_mode": "composite_expansion",
                    "query_text": "组合查询",
                    "expected_template_ids": ["template-a"],
                    "expected_direct_card_ids": ["direct-b"],
                },
            ],
        }
        rankings = [
            [
                {
                    "card_id": "direct-a",
                    "source_id": "source-a",
                    "source_scope": "local_demo_allowed",
                }
            ],
            [
                {
                    "card_id": "direct-b",
                    "source_id": "source-b",
                    "source_scope": "local_demo_allowed",
                }
            ],
        ]
        result = evaluate_rankings(
            manifest,
            rankings,
            [[], ["template-a"]],
            {"direct-a", "direct-b"},
            {"source-a", "source-b"},
        )
        self.assertEqual(result["direct_top1_primary_accuracy"], 1.0)
        self.assertEqual(result["direct_top3_expected_recall"], 1.0)
        self.assertEqual(result["composite_template_accuracy"], 1.0)
        self.assertEqual(result["composite_returned_count"], 0)
        self.assertTrue(result["source_traceability"])
        self.assertTrue(result["scope_valid"])


if __name__ == "__main__":
    unittest.main()
