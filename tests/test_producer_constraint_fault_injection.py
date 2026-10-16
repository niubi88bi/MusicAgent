"""Offline contracts for controlled Producer constraint fault injection."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.evaluate_producer_constraint_fault_injection import (
    build_blind_card,
    load_manifest,
    render_blind_review,
    resolve_cases,
    validate_fault_contracts,
)
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/output/evaluations").is_dir(), "private evaluation artifacts are not published")
class ProducerConstraintFaultInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.pairs = resolve_cases(cls.manifest)

    def test_four_fault_families_have_clean_bases_and_expected_hard_faults(self) -> None:
        contracts = validate_fault_contracts(self.manifest)
        self.assertEqual(len(contracts), 4)
        self.assertEqual(
            {row["fault_family"] for row in contracts},
            {
                "wrong_vocal_gender",
                "forbidden_rap_mc_slogans",
                "vocal_material_in_instrumental",
                "hard_fixed_bpm",
            },
        )
        self.assertTrue(all(not row["base_hard_violation_codes"] for row in contracts))
        self.assertTrue(all(row["injected_hard_violation_codes"] for row in contracts))

    def test_each_injected_plan_contains_only_hard_trigger_rows(self) -> None:
        for source_case, fault in self.pairs:
            constraints = extract_explicit_constraints(source_case.user_request_text)
            violations = check_plan_constraints(
                fault.injected_plan,
                user_text=source_case.user_request_text,
                constraints=constraints,
            )
            hard, advisory = split_constraint_violations(violations)
            self.assertEqual(
                [row.code for row in hard], fault.expected_injected_hard_codes
            )
            self.assertEqual(advisory, [])

    def test_blind_packet_compares_clean_base_with_result_without_method_leak(self) -> None:
        source_case, fault = self.pairs[0]
        card, key = build_blind_card(source_case, fault, fault.base_plan)
        visible = json.dumps(card, ensure_ascii=False) + render_blind_review([card])
        for leaked in ("fault", "injected", "repaired", "故障", "注入", "修复", "RAG"):
            self.assertNotIn(leaked, visible)
        self.assertEqual(set(key.values()), {"base", "repaired"})


if __name__ == "__main__":
    unittest.main()
