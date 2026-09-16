import json
import unittest
from pathlib import Path

from scripts.validate_style_template_contract import _read_jsonl, validate


class StyleTemplateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.cards = _read_jsonl(cls.root / "data/miaoxiang/style_templates.v2.jsonl")
        cls.sources = _read_jsonl(cls.root / "data/miaoxiang/source_catalog.jsonl")
        cls.report = validate(cls.cards, cls.sources, schema_implementation_present=False)

    def test_current_draft_cards_pass_shadow_contract(self):
        self.assertTrue(self.report["valid"], self.report["errors"])
        self.assertEqual(self.report["card_count"], 17)
        self.assertEqual(self.report["error_count"], 0)
        self.assertFalse(self.report["schema_implementation_present"])

    def test_role_scope_and_status_boundaries(self):
        composite = next(card for card in self.cards if card["id"] == "style-dynamic-sad-folk-pop-remix-v2")
        self.assertEqual(composite["card_role"], "composite_query_template")
        self.assertEqual(composite["retrieval_scope"], "query_expansion_only")
        self.assertTrue(all(card["review_status"] == "draft" for card in self.cards))
        self.assertFalse(self.report["review_status_changed"])
        self.assertFalse(self.report["production_index_changed"])

    def test_invalid_composite_scope_is_rejected(self):
        cards = json.loads(json.dumps(self.cards, ensure_ascii=False))
        composite = next(card for card in cards if card["id"] == "style-dynamic-sad-folk-pop-remix-v2")
        composite["retrieval_scope"] = "direct_candidate"
        report = validate(cards, self.sources, schema_implementation_present=False)
        self.assertFalse(report["valid"])
        self.assertTrue(any("composite_must_be_query_expansion_only" in issue for row in report["errors"] for issue in row["issues"]))

    def test_revised_voice_fields_are_complete(self):
        revised_ids = {
            "style-dynamic-vina-house-v2",
            "style-dynamic-chinese-bounce-v2",
            "style-dynamic-sad-slap-house-v2",
            "style-edm-techno-v2",
            "style-edm-future-bass-v2",
            "style-guofeng-zen-ambient-v2",
        }
        revised = [row for row in self.report["card_results"] if row["card_id"] in revised_ids]
        self.assertEqual(len(revised), 6)
        self.assertTrue(all(row["voice_fields_complete"] for row in revised))
        self.assertTrue(all(row["valid"] for row in revised))


if __name__ == "__main__":
    unittest.main()
