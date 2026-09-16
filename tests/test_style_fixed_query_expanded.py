import json
import unittest
from pathlib import Path

from scripts.evaluate_style_fixed_queries import _read_jsonl, evaluate


class StyleFixedQueryExpandedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.cards = _read_jsonl(cls.root / "data/miaoxiang/style_templates.v2.jsonl")
        cls.sources = _read_jsonl(cls.root / "data/miaoxiang/source_catalog.jsonl")
        cls.manifest = json.loads(
            (cls.root / "data/miaoxiang/style_fixed_queries.v1.3.json").read_text(encoding="utf-8")
        )
        cls.result = evaluate(cls.cards, cls.sources, cls.manifest)

    def test_holdout_manifest_has_twenty_unique_queries(self):
        query_ids = [query["query_id"] for query in self.manifest["queries"]]
        self.assertEqual(len(query_ids), 20)
        self.assertEqual(len(query_ids), len(set(query_ids)))

    def test_all_expected_cards_exist_and_are_traceable(self):
        card_ids = {card["id"] for card in self.cards}
        source_ids = {source["source_id"] for source in self.sources}
        for query in self.manifest["queries"]:
            self.assertTrue(set(query["expected_card_ids"]).issubset(card_ids))
        self.assertTrue(self.result["top1_source_traceability"])
        self.assertEqual(self.result["card_count"], 17)
        self.assertEqual(self.result["source_count"], len(source_ids))

    def test_holdout_is_evidence_not_production_admission(self):
        self.assertFalse(self.result["production_index_changed"])
        self.assertEqual(self.result["draft_card_count"], 17)
        self.assertEqual(self.result["query_count"], 20)


if __name__ == "__main__":
    unittest.main()
