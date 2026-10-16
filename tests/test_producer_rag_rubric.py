"""Offline contracts for the Producer style-RAG review rubric v2 experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agents.schemas import ProductionPlan
from scripts.evaluate_producer_rag_gated_ab import build_blind_card
from scripts.evaluate_producer_rag_rubric import (
    load_manifest,
    render_rubric_review,
    validate_gate_contracts,
)


PLAN = {
    "style_prompt": "A restrained, executable production direction with clear dynamics.",
    "tags": ["restrained", "clear", "dynamic"],
    "vocal_direction": "Follow the user's explicit vocal constraint.",
    "arrangement_note": "Keep the arrangement focused on the requested style.",
}


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/miaoxiang/style_templates.v2.jsonl").is_file(), "private style-card dataset is not published")
class ProducerRAGRubricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()

    def test_manifest_freezes_six_distinct_risk_cases(self) -> None:
        self.assertEqual(len(self.manifest.cases), 6)
        self.assertEqual(len({case.case_id for case in self.manifest.cases}), 6)
        self.assertEqual(len({case.risk_focus for case in self.manifest.cases}), 6)

    def test_all_cases_pass_single_style_gate_contracts(self) -> None:
        contracts = validate_gate_contracts(self.manifest)
        self.assertEqual(len(contracts), 6)
        self.assertTrue(all(row["action"] == "single_style" for row in contracts))
        self.assertTrue(all(len(row["candidate_ids"]) == 1 for row in contracts))
        self.assertTrue(all(row["prompts_identical"] is False for row in contracts))

    def test_blind_packet_hides_variant_identity_and_risk_focus(self) -> None:
        case = self.manifest.cases[0]
        plans = {
            "baseline": ProductionPlan.model_validate(PLAN),
            "gated": ProductionPlan.model_validate(
                {**PLAN, "arrangement_note": "Use the selected style reference."}
            ),
        }
        card, key = build_blind_card(case, plans)
        visible = json.dumps(card, ensure_ascii=False) + render_rubric_review([card])
        self.assertNotIn("baseline", visible)
        self.assertNotIn("gated", visible)
        self.assertNotIn("risk_focus", visible)
        self.assertNotIn("RAG", visible)
        self.assertEqual(set(key.values()), {"baseline", "gated"})

    def test_review_packet_contains_clarified_rubric(self) -> None:
        case = self.manifest.cases[0]
        plans = {
            "baseline": ProductionPlan.model_validate(PLAN),
            "gated": ProductionPlan.model_validate(PLAN),
        }
        card, _ = build_blind_card(case, plans)
        review = render_rubric_review([card])
        self.assertIn("不自动算扩写", review)
        self.assertIn("不可替换的强制实现", review)
        self.assertIn("mix voice / chest-head mix", review)
        self.assertIn("决定性问题", review)


if __name__ == "__main__":
    unittest.main()
