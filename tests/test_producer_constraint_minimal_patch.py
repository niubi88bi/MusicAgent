"""Offline contracts for fail-closed minimal-diff Producer repair."""

from __future__ import annotations

import unittest
from pathlib import Path

from agents.schemas import ProductionPlan
from scripts.evaluate_producer_constraint_fault_injection import (
    load_manifest,
    resolve_cases,
)
from scripts.evaluate_producer_constraint_minimal_patch import (
    build_blind_card,
    build_fragment_messages,
    plan_deterministic_stage,
    preview_case,
)
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)
from tools.rag.producer_constraint_patch import (
    FragmentPatch,
    apply_fragment_patch,
    unresolved_location,
    validate_patched_plan,
)


class ProducerConstraintMinimalPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(
            Path("data/miaoxiang/producer_constraint_local_fault_injection.v2.json")
        )
        cls.pairs = resolve_cases(cls.manifest)
        cls.by_id = {fault.case_id: (source, fault) for source, fault in cls.pairs}

    def test_dry_run_routes_three_rules_and_one_fragment_call(self) -> None:
        previews = [preview_case(source, fault) for source, fault in self.pairs]
        actions = {row["case_id"]: row["action"] for row in previews}
        self.assertEqual(
            actions,
            {
                "lf01-wrong-vocal-gender": "deterministic_only",
                "lf02-forbidden-rap-mc-slogans": "llm_fragment_once",
                "lf03-vocal-material-in-instrumental": "deterministic_only",
                "lf04-hard-fixed-bpm": "deterministic_only",
            },
        )

    def test_deterministic_cases_restore_frozen_bases_exactly(self) -> None:
        for case_id in (
            "lf01-wrong-vocal-gender",
            "lf03-vocal-material-in-instrumental",
            "lf04-hard-fixed-bpm",
        ):
            source, fault = self.by_id[case_id]
            repaired, patches, unresolved = plan_deterministic_stage(
                fault.injected_plan,
                user_text=source.user_request_text,
            )
            self.assertEqual(unresolved, [])
            self.assertTrue(patches)
            self.assertEqual(repaired, fault.base_plan)
            self.assertTrue(all(row.strategy == "deterministic" for row in patches))

    def test_rap_case_localizes_one_fragment_without_exposing_other_fields(self) -> None:
        source, fault = self.by_id["lf02-forbidden-rap-mc-slogans"]
        intermediate, patches, unresolved = plan_deterministic_stage(
            fault.injected_plan,
            user_text=source.user_request_text,
        )
        self.assertEqual(patches, [])
        self.assertEqual(len(unresolved), 1)
        location = unresolved_location(intermediate, unresolved[0])
        self.assertIsNotNone(location)
        assert location is not None
        self.assertEqual(location.field, "arrangement_note")
        self.assertIn("Rap", location.original_fragment)
        messages = build_fragment_messages(
            user_text=source.user_request_text,
            field_value=intermediate.arrangement_note,
            location=location,
            violation=unresolved[0],
        )
        visible = "\n".join(row["content"] for row in messages)
        self.assertNotIn(intermediate.style_prompt, visible)
        self.assertNotIn(intermediate.vocal_direction, visible)

    def test_exact_patch_preserves_unapproved_fields(self) -> None:
        source, fault = self.by_id["lf02-forbidden-rap-mc-slogans"]
        constraints = extract_explicit_constraints(source.user_request_text)
        violations = check_plan_constraints(
            fault.injected_plan,
            user_text=source.user_request_text,
            constraints=constraints,
        )
        hard, _ = split_constraint_violations(violations)
        location = unresolved_location(fault.injected_plan, hard[0])
        assert location is not None
        patch = FragmentPatch(
            field=location.field,
            original_fragment=location.original_fragment,
            replacement_fragment="桥段降低鼓组密度",
            violation_code=hard[0].code,
            concept=hard[0].concept,
            strategy="llm_fragment",
        )
        repaired = apply_fragment_patch(fault.injected_plan, patch)
        validation = validate_patched_plan(
            fault.injected_plan,
            repaired,
            patches=[patch],
            user_text=source.user_request_text,
            constraints=constraints,
        )
        self.assertTrue(validation.accepted)
        self.assertEqual(validation.changed_fields, ["arrangement_note"])
        self.assertEqual(repaired.style_prompt, fault.injected_plan.style_prompt)
        self.assertEqual(repaired.tags, fault.injected_plan.tags)
        self.assertEqual(repaired.vocal_direction, fault.injected_plan.vocal_direction)

    def test_validator_rejects_new_registered_concept(self) -> None:
        source, fault = self.by_id["lf02-forbidden-rap-mc-slogans"]
        constraints = extract_explicit_constraints(source.user_request_text)
        payload = fault.injected_plan.model_dump(mode="json")
        payload["arrangement_note"] = payload["arrangement_note"].replace(
            "桥段加入短Rap、MC callouts和采样口号",
            "桥段加入钢琴并降低鼓组密度",
            1,
        )
        repaired = ProductionPlan.model_validate(payload)
        patch = FragmentPatch(
            field="arrangement_note",
            original_fragment="桥段加入短Rap、MC callouts和采样口号",
            replacement_fragment="桥段加入钢琴并降低鼓组密度",
            violation_code="forbidden_term_present",
            concept="rap_or_mc",
            strategy="llm_fragment",
        )
        validation = validate_patched_plan(
            fault.injected_plan,
            repaired,
            patches=[patch],
            user_text=source.user_request_text,
            constraints=constraints,
        )
        self.assertFalse(validation.accepted)
        self.assertEqual(validation.reason, "new_registered_concepts")
        self.assertIn("piano", validation.new_registered_concepts)

    def test_validator_rejects_extra_change_inside_allowed_field(self) -> None:
        source, fault = self.by_id["lf02-forbidden-rap-mc-slogans"]
        constraints = extract_explicit_constraints(source.user_request_text)
        original_fragment = "桥段加入短Rap、MC callouts和采样口号"
        replacement_fragment = "桥段降低鼓组密度"
        patch = FragmentPatch(
            field="arrangement_note",
            original_fragment=original_fragment,
            replacement_fragment=replacement_fragment,
            violation_code="forbidden_term_present",
            concept="rap_or_mc",
            strategy="llm_fragment",
        )
        payload = fault.injected_plan.model_dump(mode="json")
        payload["arrangement_note"] = (
            payload["arrangement_note"]
            .replace(original_fragment, replacement_fragment, 1)
            .replace("最后回到简洁的收束", "最后加入钢琴并收束", 1)
        )
        tampered = ProductionPlan.model_validate(payload)
        validation = validate_patched_plan(
            fault.injected_plan,
            tampered,
            patches=[patch],
            user_text=source.user_request_text,
            constraints=constraints,
        )
        self.assertFalse(validation.accepted)
        self.assertEqual(validation.reason, "patch_reconstruction_mismatch")

    def test_patch_application_requires_one_exact_source_fragment(self) -> None:
        _, fault = self.by_id["lf04-hard-fixed-bpm"]
        patch = FragmentPatch(
            field="style_prompt",
            original_fragment="not present",
            replacement_fragment="around 123 BPM",
            violation_code="exact_bpm_written_as_hard_requirement",
            concept="bpm",
            strategy="deterministic",
        )
        with self.assertRaisesRegex(ValueError, "must_match_once"):
            apply_fragment_patch(fault.injected_plan, patch)

    def test_blind_card_uses_neutral_id_and_hides_method_identity(self) -> None:
        source, fault = self.by_id["lf02-forbidden-rap-mc-slogans"]
        card, key = build_blind_card(source, fault, fault.base_plan)
        visible = str(card)
        for leaked in ("minimal", "base", "repair", "fault", "修复", "故障"):
            self.assertNotIn(leaked, visible)
        self.assertEqual(card["case_id"], "l02-rain-night-bus")
        self.assertEqual(set(key.values()), {"base", "minimal"})


if __name__ == "__main__":
    unittest.main()
