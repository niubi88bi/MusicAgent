"""Offline behavior tests for the deterministic style-RAG shadow gate."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.evaluate_style_rag_gate import (
    HOLDOUT_PATH,
    build_report,
    load_acceptance,
    load_manifest,
)
from tools.rag.style_retrieval_policy import decide_style_retrieval, parse_constraints


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/miaoxiang/style_templates.v2.jsonl").is_file(), "private style-card dataset is not published")
class StyleRAGGateTests(unittest.TestCase):
    def test_development_and_acceptance_contracts_pass(self) -> None:
        report = build_report(load_acceptance(), load_manifest(HOLDOUT_PATH))

        self.assertEqual(report["development"], {"count": 8, "passed": 8})
        self.assertEqual(report["acceptance"]["count"], 12)
        self.assertEqual(report["acceptance"]["passed"], 12)
        self.assertFalse(report["acceptance"]["independent_holdout"])
        self.assertEqual(report["holdout"]["count"], 12)
        self.assertEqual(report["aggregate"]["runtime_injection_enabled_count"], 0)

    def test_broad_and_low_confidence_terms_do_not_authorize_rag(self) -> None:
        broad = decide_style_retrieval("做一首浩室，晚上开车听")
        low = decide_style_retrieval("做一首车载DJ")

        self.assertEqual((broad.action, broad.reason), ("no_rag", "broad_style_only"))
        self.assertEqual(
            (low.action, low.reason),
            ("no_rag", "low_confidence_alias_without_corroboration"),
        )

    def test_low_alias_with_distinctive_evidence_can_authorize_one_card(self) -> None:
        decision = decide_style_retrieval("慢摇，四拍和弹性贝斯，中文女声")

        self.assertEqual(decision.action, "single_style")
        self.assertEqual(
            decision.candidate_ids, ["style-dynamic-chinese-bounce-v2"]
        )

    def test_rap_word_does_not_match_inside_trap(self) -> None:
        constraints = parse_constraints("Guofeng Trap，纯音乐")

        self.assertEqual(constraints.rap, "unspecified")
        self.assertEqual(constraints.instrumental, "required")

    def test_optional_rap_returns_paired_rap_and_non_rap_cards(self) -> None:
        decision = decide_style_retrieval("国风电子，说唱可有可无")

        self.assertEqual(decision.action, "multi_candidate")
        self.assertEqual(
            decision.candidate_ids,
            [
                "style-guofeng-electronic-fusion-v2",
                "style-rap-cyber-zen-v2",
            ],
        )

    def test_explicit_supported_and_unsupported_mix_fails_closed(self) -> None:
        decision = decide_style_retrieval("Deep House 加一点朋克摇滚")

        self.assertEqual(decision.action, "no_rag")
        self.assertEqual(decision.reason, "partially_unsupported_multiple_style")
        self.assertFalse(decision.runtime_injection_allowed)

    def test_composite_returns_two_cards_and_guidance_not_template(self) -> None:
        decision = decide_style_retrieval("苦情民谣电音，烟嗓夜路")

        self.assertEqual(decision.action, "multi_candidate")
        self.assertEqual(len(decision.candidate_ids), 2)
        self.assertNotIn(decision.composite_template_id, decision.candidate_ids)
        self.assertIsNotNone(decision.composite_guidance)


if __name__ == "__main__":
    unittest.main()
