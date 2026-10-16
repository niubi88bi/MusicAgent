"""Offline contracts for the conservative-gate Producer blind experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agents.schemas import ProductionPlan
from scripts.evaluate_producer_rag_gated_ab import (
    blind_mapping,
    build_blind_card,
    build_gate_context,
    build_variant_messages,
    load_cases,
    render_blind_review,
)
from tools.rag.style_retrieval_policy import decide_style_retrieval


PLAN = {
    "style_prompt": "Restrained electronic folk, warm vocal, gradual dynamics.",
    "tags": ["restrained", "electronic folk", "warm"],
    "vocal_direction": "Keep the delivery close and natural.",
    "arrangement_note": "Expand only slightly in the chorus.",
}


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/miaoxiang/style_templates.v2.jsonl").is_file(), "private style-card dataset is not published")
class ProducerRAGGatedABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_cases()

    def test_manifest_has_four_distinct_required_types(self) -> None:
        self.assertEqual(len(self.manifest.cases), 4)
        self.assertEqual(
            {case.case_type for case in self.manifest.cases},
            {
                "clear_single_style",
                "named_composite",
                "emotion_only",
                "hard_vocal_constraint",
            },
        )

    def test_frozen_cases_match_gate_contract(self) -> None:
        for case in self.manifest.cases:
            with self.subTest(case=case.case_id):
                decision = decide_style_retrieval(case.user_request_text)
                self.assertEqual(decision.action, case.expected_action)
                self.assertEqual(decision.candidate_ids, case.expected_candidate_ids)

    def test_no_rag_pair_has_identical_prompts(self) -> None:
        case = next(
            item for item in self.manifest.cases if item.expected_action == "no_rag"
        )
        decision = decide_style_retrieval(case.user_request_text)
        context = build_gate_context(decision)
        baseline = build_variant_messages(
            case, variant="baseline", decision=decision, context=context
        )
        gated = build_variant_messages(
            case, variant="gated", decision=decision, context=context
        )
        self.assertEqual(context, [])
        self.assertEqual(baseline, gated)

    def test_positive_gate_changes_only_experimental_variant(self) -> None:
        case = next(
            item
            for item in self.manifest.cases
            if item.expected_action == "single_style"
        )
        decision = decide_style_retrieval(case.user_request_text)
        context = build_gate_context(decision)
        baseline = build_variant_messages(
            case, variant="baseline", decision=decision, context=context
        )
        gated = build_variant_messages(
            case, variant="gated", decision=decision, context=context
        )
        self.assertNotEqual(baseline, gated)
        self.assertEqual(
            [row["card_id"] for row in context], case.expected_candidate_ids
        )
        self.assertNotIn("retrieval_score", context[0])

    def test_blind_packet_hides_variant_identity(self) -> None:
        case = self.manifest.cases[0]
        plans = {
            "baseline": ProductionPlan.model_validate(PLAN),
            "gated": ProductionPlan.model_validate(
                {**PLAN, "arrangement_note": "Use the selected reference carefully."}
            ),
        }
        card, key = build_blind_card(case, plans)
        visible = json.dumps(card, ensure_ascii=False) + render_blind_review([card])
        self.assertNotIn("baseline", visible)
        self.assertNotIn("gated", visible)
        self.assertNotIn("RAG", visible)
        self.assertEqual(set(key), {"X", "Y"})
        self.assertEqual(set(key.values()), {"baseline", "gated"})
        self.assertEqual(blind_mapping(case.case_id), blind_mapping(case.case_id))


if __name__ == "__main__":
    unittest.main()
