"""Offline contracts for repair-only-on-violation Producer post-checking."""

from __future__ import annotations

import json
import unittest

from scripts.evaluate_producer_constraint_postcheck import (
    blind_mapping,
    build_blind_card,
    build_repair_messages,
    load_manifest,
    render_blind_review,
    resolve_cases,
    validate_postcheck_contracts,
)
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
)


class ProducerConstraintPostcheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.cases = {
            source.case_id: (source, frozen)
            for source, frozen in resolve_cases(cls.manifest)
        }

    def test_frozen_contract_routes_two_pass_through_and_one_repair(self) -> None:
        contracts = validate_postcheck_contracts(self.manifest)
        actions = {row["case_id"]: row["action"] for row in contracts}
        self.assertEqual(
            actions,
            {
                "r01-deep-house-mix-voice-flex-tempo": "pass_through",
                "r05-vina-house-female-vocal-override": "pass_through",
                "r06-sad-slap-house-no-piano": "repair_once",
            },
        )
        self.assertEqual(sum(row["planned_llm_calls"] for row in contracts), 1)

    def test_pass_through_uses_exact_frozen_plan_without_call_contract(self) -> None:
        for case_id in (
            "r01-deep-house-mix-voice-flex-tempo",
            "r05-vina-house-female-vocal-override",
        ):
            source, frozen = self.cases[case_id]
            violations = check_plan_constraints(
                frozen.initial_plan,
                user_text=source.user_request_text,
                constraints=extract_explicit_constraints(source.user_request_text),
            )
            self.assertEqual(violations, [])

    def test_repair_message_is_postcheck_only_and_bounded_to_found_conflicts(self) -> None:
        source, frozen = self.cases["r06-sad-slap-house-no-piano"]
        violations = check_plan_constraints(
            frozen.initial_plan,
            user_text=source.user_request_text,
            constraints=extract_explicit_constraints(source.user_request_text),
        )
        messages = build_repair_messages(source, frozen.initial_plan, violations)
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertIn("确定性后检发现", messages[-1]["content"])
        self.assertIn("required_term_missing", messages[-1]["content"])
        self.assertIn("forbidden_term_present", messages[-1]["content"])
        self.assertIn("unrequested_vocal_gender", messages[-1]["content"])
        self.assertNotIn("优先级高于风格参考卡", messages[1]["content"])

    def test_blind_packet_hides_initial_and_postcheck_identity(self) -> None:
        source, frozen = self.cases["r06-sad-slap-house-no-piano"]
        card, key = build_blind_card(
            source,
            initial_plan=frozen.initial_plan,
            final_plan=frozen.initial_plan.model_copy(
                update={"arrangement_note": "木吉他承担伤感动机，全程不使用钢琴。"}
            ),
        )
        visible = json.dumps(card, ensure_ascii=False) + render_blind_review([card])
        self.assertNotIn("postcheck", visible.lower())
        self.assertNotIn("gated", visible.lower())
        self.assertNotIn("RAG", visible)
        self.assertEqual(set(key.values()), {"initial", "postcheck"})
        self.assertEqual(set(blind_mapping(source.case_id)), {"X", "Y"})


if __name__ == "__main__":
    unittest.main()
