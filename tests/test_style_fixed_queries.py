import json
import unittest
from pathlib import Path

from scripts.evaluate_style_fixed_queries import evaluate, _read_jsonl


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/miaoxiang/style_templates.v2.jsonl").is_file(), "private style-card dataset is not published")
class StyleFixedQueriesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.manifest = json.loads(
            (self.root / "data/miaoxiang/style_fixed_queries.v1.json").read_text(encoding="utf-8")
        )
        self.manifest_v11 = json.loads(
            (self.root / "data/miaoxiang/style_fixed_queries.v1.1.json").read_text(encoding="utf-8")
        )
        self.cards = _read_jsonl(self.root / "data/miaoxiang/style_templates.v2.jsonl")
        self.sources = _read_jsonl(self.root / "data/miaoxiang/source_catalog.jsonl")

    def test_manifest_covers_all_draft_cards_as_expected_or_primary(self):
        covered = {
            card_id
            for query in self.manifest["queries"]
            for card_id in query["expected_card_ids"]
        }
        self.assertEqual(covered, {card["id"] for card in self.cards})
        self.assertEqual(len(self.manifest["queries"]), 8)

    def test_lexical_baseline_is_deterministic_and_traceable(self):
        result = evaluate(self.cards, self.sources, self.manifest)
        self.assertEqual(result["query_count"], 8)
        self.assertEqual(result["top1_primary_accuracy"], 1.0)
        self.assertEqual(result["top3_expected_recall"], 1.0)
        self.assertTrue(result["top1_source_traceability"])
        self.assertFalse(result["production_index_changed"])
        self.assertGreater(result["low_signal_query_count"], 0)

    def test_v11_reduces_low_signal_queries_without_changing_primary_hits(self):
        result = evaluate(self.cards, self.sources, self.manifest_v11)
        self.assertEqual(result["top1_primary_accuracy"], 1.0)
        self.assertEqual(result["top3_expected_recall"], 1.0)
        self.assertEqual(result["low_signal_query_count"], 0)
        self.assertFalse(result["production_index_changed"])


if __name__ == "__main__":
    unittest.main()
