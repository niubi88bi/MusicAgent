"""Offline contracts for hard-vs-advisory Producer constraint routing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agents.schemas import ProductionPlan
from scripts.evaluate_producer_constraint_severity_holdout import (
    build_blind_card,
    build_initial_messages,
    build_repair_messages,
    load_manifest,
    render_blind_review,
    should_repair,
    validate_holdout_contracts,
)
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)


def _plan(style: str, vocal: str, arrangement: str) -> ProductionPlan:
    return ProductionPlan(
        style_prompt=style,
        tags=["style", "mood", "demo"],
        vocal_direction=vocal,
        arrangement_note=arrangement,
    )


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/output/evaluations").is_dir(), "private evaluation artifacts are not published")
class ProducerConstraintSeverityHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.cases = {case.case_id: case for case in cls.manifest.cases}

    def test_eight_frozen_contracts_include_six_hard_and_two_soft_cases(self) -> None:
        contracts = validate_holdout_contracts(self.manifest)
        self.assertEqual(len(contracts), 8)
        self.assertEqual(
            sum(row["has_explicit_hard_contract"] for row in contracts), 6
        )
        soft = {
            row["case_id"]
            for row in contracts
            if not row["has_explicit_hard_contract"]
        }
        self.assertEqual(
            soft,
            {
                "s01-tropical-house-open-gender",
                "s02-future-bass-single-style",
            },
        )

    def test_advisory_only_gender_does_not_trigger_repair(self) -> None:
        case = self.cases["s01-tropical-house-open-gender"]
        constraints = extract_explicit_constraints(case.user_request_text)
        violations = check_plan_constraints(
            _plan(
                "Tropical House with clear female vocal and marimba.",
                "Bright female solo vocal.",
                "Keep the chorus breezy.",
            ),
            user_text=case.user_request_text,
            constraints=constraints,
        )
        hard, advisory = split_constraint_violations(violations)
        self.assertEqual(hard, [])
        self.assertIn("unrequested_vocal_gender", {row.code for row in advisory})
        self.assertFalse(should_repair(violations))

    def test_explicit_instrument_conflict_triggers_repair(self) -> None:
        case = self.cases["h04-sad-slap-guitar-pharmacy"]
        constraints = extract_explicit_constraints(case.user_request_text)
        violations = check_plan_constraints(
            _plan(
                "Sad Slap House with sad piano and deep bass.",
                "Restrained close-mic solo vocal.",
                "Open with piano and pads.",
            ),
            user_text=case.user_request_text,
            constraints=constraints,
        )
        hard, advisory = split_constraint_violations(violations)
        self.assertIn("required_term_missing", {row.code for row in hard})
        self.assertIn("forbidden_term_present", {row.code for row in hard})
        self.assertEqual(advisory, [])
        self.assertTrue(should_repair(violations))

    def test_repair_prompt_rejects_advisory_rows(self) -> None:
        case = self.cases["s01-tropical-house-open-gender"]
        constraints = extract_explicit_constraints(case.user_request_text)
        plan = _plan(
            "Tropical House with clear female vocal and marimba.",
            "Bright female solo vocal.",
            "Keep the chorus breezy.",
        )
        violations = check_plan_constraints(
            plan,
            user_text=case.user_request_text,
            constraints=constraints,
        )
        with self.assertRaisesRegex(ValueError, "hard_violations_only"):
            build_repair_messages(build_initial_messages(case), plan, violations)

    def test_blind_packet_hides_method_identity(self) -> None:
        case = self.cases["h04-sad-slap-guitar-pharmacy"]
        initial = _plan(
            "Sad Slap House with sad piano and deep bass.",
            "Restrained solo vocal.",
            "Piano intro.",
        )
        repaired = _plan(
            "Sad Slap House with acoustic guitar and deep bass.",
            "Restrained solo vocal.",
            "Acoustic guitar intro.",
        )
        card, key = build_blind_card(
            case, initial_plan=initial, repaired_plan=repaired
        )
        visible = json.dumps(card, ensure_ascii=False) + render_blind_review([card])
        self.assertNotIn("repaired", visible.lower())
        self.assertNotIn("initial", visible.lower())
        self.assertNotIn("修复", visible)
        self.assertNotIn("RAG", visible)
        self.assertEqual(set(key.values()), {"initial", "repaired"})


if __name__ == "__main__":
    unittest.main()
