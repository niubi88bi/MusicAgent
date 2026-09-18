"""Offline contracts for repeated Producer style-RAG boundary validation."""

from __future__ import annotations

import json
import unittest

from agents.schemas import ProductionPlan
from scripts.evaluate_producer_rag_boundary import (
    _render_boundary_review,
    expanded_generation_cases,
    load_manifest,
    validate_gate_contracts,
)
from scripts.evaluate_producer_rag_gated_ab import build_blind_card


PLAN = {
    "style_prompt": "Melodic electronic, gradual build, controlled chorus release.",
    "tags": ["melodic electronic", "gradual", "controlled"],
    "vocal_direction": "Keep the vocal clear and emotionally restrained.",
    "arrangement_note": "Build gradually and avoid stacking unrelated drops.",
}


class ProducerRAGBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()

    def test_manifest_expands_to_four_pairs_and_eight_calls(self) -> None:
        expanded = expanded_generation_cases(self.manifest)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(len(expanded) * 2, 8)
        self.assertEqual(
            {row.family_id.rsplit("-r", 1)[0] for row in expanded},
            {row.family_id for row in self.manifest.generation_families},
        )

    def test_all_gate_contracts_pass_before_generation(self) -> None:
        contracts = validate_gate_contracts(self.manifest)
        self.assertEqual(len(contracts["generation_families"]), 2)
        self.assertEqual(len(contracts["offline_cases"]), 2)
        self.assertTrue(
            all(row["action"] == "multi_candidate" for row in contracts["generation_families"])
        )
        self.assertTrue(
            all(row["action"] == "no_rag" for row in contracts["offline_cases"])
        )
        self.assertTrue(
            all(row["deepseek_calls"] == 0 for row in contracts["offline_cases"])
        )

    def test_blind_review_hides_variant_identity(self) -> None:
        case = expanded_generation_cases(self.manifest)[0]
        plans = {
            "baseline": ProductionPlan.model_validate(PLAN),
            "gated": ProductionPlan.model_validate(
                {**PLAN, "arrangement_note": "Use only the selected references."}
            ),
        }
        card, key = build_blind_card(case, plans)
        card["family_id"] = case.family_id.rsplit("-r", 1)[0]
        card["replicate"] = 1
        visible = json.dumps(card, ensure_ascii=False) + _render_boundary_review([card])
        self.assertNotIn("baseline", visible)
        self.assertNotIn("gated", visible)
        self.assertNotIn("RAG", visible)
        self.assertEqual(set(key.values()), {"baseline", "gated"})


if __name__ == "__main__":
    unittest.main()
