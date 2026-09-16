import json
import unittest
from pathlib import Path

from scripts.evaluate_style_fixed_queries import _read_jsonl
from scripts.evaluate_style_policy_shadow import evaluate


class StylePolicyShadowTests(unittest.TestCase):
    def test_role_and_vocal_policies(self):
        root = Path(__file__).resolve().parents[1]
        cards = _read_jsonl(root / "data/miaoxiang/style_templates.v2.jsonl")
        manifest = json.loads(
            (root / "data/miaoxiang/style_fixed_queries.v1.2.json").read_text(encoding="utf-8")
        )
        result = evaluate(cards, manifest)
        self.assertEqual(result["composite_template_top1_accuracy"], 1.0)
        self.assertEqual(result["direct_primary_top1_accuracy"], 1.0)
        self.assertEqual(result["direct_expected_top3_recall"], 1.0)
        self.assertTrue(result["direct_top3_excludes_composite"])
        self.assertFalse(result["production_index_changed"])
        self.assertFalse(result["review_status_changed"])


if __name__ == "__main__":
    unittest.main()
