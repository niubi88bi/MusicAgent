"""Offline contracts for the explicit Producer constraint guard experiment."""

from __future__ import annotations

import json
import unittest

from agents.schemas import ProductionPlan
from scripts.evaluate_producer_constraint_guard import (
    build_blind_card,
    build_guarded_messages,
    load_manifest,
    render_blind_review,
    reanalyze_existing_run,
    resolve_cases,
    validate_guard_contracts,
)
from scripts.evaluate_producer_rag_gated_ab import build_gate_context
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
)
from tools.rag.style_retrieval_policy import decide_style_retrieval


def _plan(style_prompt: str, vocal: str, arrangement: str) -> ProductionPlan:
    return ProductionPlan(
        style_prompt=style_prompt,
        tags=["style", "mood", "demo"],
        vocal_direction=vocal,
        arrangement_note=arrangement,
    )


class ProducerConstraintGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.cases = {case.case_id: case for case, _ in resolve_cases(cls.manifest)}

    def test_manifest_constraints_match_frozen_expectations(self) -> None:
        contracts = validate_guard_contracts(self.manifest)
        self.assertEqual(len(contracts), 3)
        self.assertTrue(all(row["prompts_identical"] is False for row in contracts))

    def test_checker_catches_unrequested_gender(self) -> None:
        case = self.cases["r01-deep-house-mix-voice-flex-tempo"]
        constraints = extract_explicit_constraints(case.user_request_text)
        violations = check_plan_constraints(
            _plan(
                "Deep House with a gentle Chinese male vocal.",
                "Male solo using chest-head mix.",
                "Keep the chorus restrained.",
            ),
            user_text=case.user_request_text,
            constraints=constraints,
        )
        self.assertIn("unrequested_vocal_gender", {row.code for row in violations})

    def test_checker_catches_secondary_genre(self) -> None:
        case = self.cases["r05-vina-house-female-vocal-override"]
        constraints = extract_explicit_constraints(case.user_request_text)
        violations = check_plan_constraints(
            _plan(
                "Vina House with city-pop-tinged production and female vocal.",
                "Bright female vocal, no male voice.",
                "Keep it light and fast.",
            ),
            user_text=case.user_request_text,
            constraints=constraints,
        )
        self.assertIn("unrequested_secondary_genre", {row.code for row in violations})

    def test_checker_catches_forbidden_and_missing_instrument(self) -> None:
        case = self.cases["r06-sad-slap-house-no-piano"]
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
        codes = {row.code for row in violations}
        self.assertIn("forbidden_term_present", codes)
        self.assertIn("required_term_missing", codes)

    def test_checker_does_not_treat_negated_terms_as_usage(self) -> None:
        female_case = self.cases["r05-vina-house-female-vocal-override"]
        female_constraints = extract_explicit_constraints(female_case.user_request_text)
        female_violations = check_plan_constraints(
            _plan(
                "Vina House with a clean female vocal and no raspy male voice.",
                "清亮女声独唱，禁止烟嗓男声。",
                "Keep the morning mood light.",
            ),
            user_text=female_case.user_request_text,
            constraints=female_constraints,
        )
        self.assertEqual(female_violations, [])

        guitar_case = self.cases["r06-sad-slap-house-no-piano"]
        guitar_constraints = extract_explicit_constraints(guitar_case.user_request_text)
        guitar_violations = check_plan_constraints(
            _plan(
                "Sad Slap House with acoustic guitar and deep bass; avoid piano.",
                "Restrained close-mic solo vocal.",
                "木吉他承担伤感动机，全程避免钢琴音色。",
            ),
            user_text=guitar_case.user_request_text,
            constraints=guitar_constraints,
        )
        self.assertEqual(guitar_violations, [])

    def test_guarded_prompt_appends_priority_contract(self) -> None:
        case = self.cases["r06-sad-slap-house-no-piano"]
        decision = decide_style_retrieval(case.user_request_text)
        context = build_gate_context(decision)
        messages = build_guarded_messages(
            case,
            decision=decision,
            context=context,
            constraints=extract_explicit_constraints(case.user_request_text),
        )
        self.assertIn("优先级高于风格参考卡", messages[1]["content"])
        self.assertIn("acoustic_guitar", messages[1]["content"])
        self.assertIn("piano", messages[1]["content"])

    def test_blind_packet_hides_method_identity(self) -> None:
        case = self.cases["r06-sad-slap-house-no-piano"]
        plans = {
            "gated": _plan("Sad Slap House with piano.", "Solo vocal.", "Piano intro."),
            "guarded": _plan(
                "Sad Slap House with acoustic guitar.",
                "Solo vocal.",
                "Acoustic guitar intro.",
            ),
        }
        card, key = build_blind_card(case, plans)
        visible = json.dumps(card, ensure_ascii=False) + render_blind_review([card])
        self.assertNotIn("gated", visible)
        self.assertNotIn("guarded", visible)
        self.assertNotIn("RAG", visible)
        self.assertEqual(set(key.values()), {"gated", "guarded"})

    def test_reanalysis_api_is_available_without_runtime_connection(self) -> None:
        self.assertTrue(callable(reanalyze_existing_run))


if __name__ == "__main__":
    unittest.main()
