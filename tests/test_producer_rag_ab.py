"""Offline contract tests for the Producer RAG-off/on experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.producer_agent import ProducerAgent
from scripts.evaluate_producer_rag_ab import (
    CASES_PATH,
    _compact_card,
    blind_mapping,
    build_blind_card,
    build_producer_messages,
    load_cases,
    render_blind_review,
    variant_order,
)
from tools.rag.style_index import load_index_inputs, select_local_demo_cards


PLAN = {
    "style_prompt": (
        "Deep house, restrained midnight mood, electric piano and warm bass, "
        "close male vocal, gradual chorus expansion."
    ),
    "tags": ["deep house", "midnight", "warm bass"],
    "vocal_direction": "Close and restrained male vocal.",
    "arrangement_note": "Keep the verse sparse and widen the chorus.",
}


class CaptureLLM:
    def __init__(self) -> None:
        self.messages: Sequence[Mapping[str, str]] | None = None
        self.kwargs: dict[str, Any] = {}

    def chat_json(
        self, messages: Sequence[Mapping[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        self.messages = messages
        self.kwargs = kwargs
        return PLAN


@unittest.skipUnless(CASES_PATH.is_file(), "private Producer RAG cases are not published")
class ProducerRAGABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_cases()
        cls.case = cls.manifest.cases[0]

    def test_fixed_manifest_has_required_case_mix(self) -> None:
        self.assertEqual(len(self.manifest.cases), 8)
        counts: dict[str, int] = {}
        for case in self.manifest.cases:
            counts[case.case_type] = counts.get(case.case_type, 0) + 1
        self.assertEqual(
            counts,
            {
                "clear_single_style": 3,
                "ambiguous_neighbor": 2,
                "named_composite": 1,
                "emotion_only": 1,
                "out_of_scope": 1,
            },
        )

    def test_rag_off_prompt_matches_current_producer_agent(self) -> None:
        llm = CaptureLLM()
        ProducerAgent(llm).run(self.case.requirement, self.case.lyric_draft)

        self.assertEqual(
            list(llm.messages or []),
            build_producer_messages(self.case.requirement, self.case.lyric_draft),
        )

    def test_rag_on_only_appends_context_to_user_message(self) -> None:
        off = build_producer_messages(self.case.requirement, self.case.lyric_draft)
        on = build_producer_messages(
            self.case.requirement,
            self.case.lyric_draft,
            rag_context=[{"card_id": "card-1", "summary": "reference"}],
        )

        self.assertEqual(off[0], on[0])
        self.assertTrue(on[1]["content"].startswith(off[1]["content"]))
        self.assertIn("候选不匹配时必须忽略", on[1]["content"])

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/miaoxiang/style_templates.v2.jsonl").is_file(), "private style-card dataset is not published")
    def test_compact_card_excludes_generation_and_license_fields(self) -> None:
        cards, sources, policy = load_index_inputs()
        direct, _, blocked = select_local_demo_cards(cards, sources, policy)
        self.assertFalse(blocked)
        payload = _compact_card(direct[0], score=0.75)
        serialized = json.dumps(payload, ensure_ascii=False)

        for forbidden in (
            "prompt_template",
            "license_or_usage_note",
            "scene_tags",
            "backend_tags",
            "official_facts",
        ):
            self.assertNotIn(forbidden, payload)
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(payload["source_scope"], "local_demo_allowed")
        self.assertEqual(payload["heartmula_validation"], "unverified")

    def test_blind_packet_does_not_expose_variant_identity(self) -> None:
        from agents.schemas import ProductionPlan

        plans = {
            "rag_off": ProductionPlan.model_validate(PLAN),
            "rag_on": ProductionPlan.model_validate(
                {**PLAN, "arrangement_note": "Use one retrieved reference only."}
            ),
        }
        card, key = build_blind_card(self.case, plans)
        review = render_blind_review([card])
        visible = json.dumps(card, ensure_ascii=False) + review

        self.assertNotIn("rag_off", visible)
        self.assertNotIn("rag_on", visible)
        self.assertNotIn("RAG", review)
        self.assertIn(self.case.lyric_draft.lyrics, review)
        self.assertEqual(set(key), {"X", "Y"})
        self.assertEqual(set(key.values()), {"rag_off", "rag_on"})

    def test_orders_and_blind_mapping_are_repeatable(self) -> None:
        self.assertEqual(variant_order(self.case.case_id), variant_order(self.case.case_id))
        self.assertEqual(blind_mapping(self.case.case_id), blind_mapping(self.case.case_id))


if __name__ == "__main__":
    unittest.main()
