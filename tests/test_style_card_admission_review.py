import json
import unittest
from pathlib import Path

from scripts.prepare_style_card_admission_review import _read_json, _read_jsonl, prepare


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/miaoxiang/style_templates.v2.jsonl").is_file(), "private style-card dataset is not published")
class StyleCardAdmissionReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.cards = _read_jsonl(cls.root / "data/miaoxiang/style_templates.v2.jsonl")
        cls.sources = _read_jsonl(cls.root / "data/miaoxiang/source_catalog.jsonl")
        cls.embedding = _read_json(
            cls.root / "data/output/evaluations/style-embedding-v1.1/result.json"
        )
        cls.policy = _read_json(
            cls.root / "data/output/evaluations/style-policy-shadow-v1.2/result.json"
        )
        cls.ai_audit = _read_json(cls.root / "data/miaoxiang/style_templates.v2.ai-audit.json")
        cls.admission = _read_json(cls.root / "data/miaoxiang/style_templates.v2.admission-audit.json")
        cls.packet = prepare(
            cls.cards,
            cls.sources,
            cls.embedding,
            cls.policy,
            cls.ai_audit,
            cls.admission,
        )

    def test_covers_every_card_once_and_sources_resolve(self):
        card_ids = [card["card_id"] for card in self.packet["cards"]]
        self.assertEqual(len(card_ids), 17)
        self.assertEqual(len(card_ids), len(set(card_ids)))
        self.assertEqual(self.packet["missing_source_card_ids"], [])
        self.assertEqual(self.packet["ungrouped_card_ids"], [])

    def test_packet_is_non_mutating_and_mirrors_human_decisions(self):
        self.assertFalse(self.packet["review_status_mutated"])
        self.assertFalse(self.packet["production_index_changed"])
        self.assertTrue(self.packet["human_review_recorded"])
        self.assertEqual(
            {card["human_decision"] for card in self.packet["cards"]},
            {"keep", "revise"},
        )
        self.assertTrue(all(card["human_notes"] for card in self.packet["cards"]))
        self.assertTrue(all(card["review_status"] == "approved" for card in self.packet["cards"]))

    def test_composite_card_is_not_direct_candidate(self):
        card = next(
            card
            for card in self.packet["cards"]
            if card["card_id"] == "style-dynamic-sad-folk-pop-remix-v2"
        )
        self.assertEqual(card["card_role"], "composite_query_template")
        self.assertEqual(card["retrieval_scope"], "query_expansion_only")
        self.assertFalse(card["direct_candidate"])

    def test_retrieval_term_classes_keep_aliases_and_scenes_below_core_terms(self):
        from scripts.evaluate_style_fixed_queries import _score

        vina = next(card for card in self.cards if card["id"] == "style-dynamic-vina-house-v2")
        core_score, _ = _score({"keywords": ["Vina House"]}, vina)
        alias_score, _ = _score({"keywords": ["土嗨"]}, vina)
        scene_score, _ = _score({"keywords": ["短视频BGM"]}, vina)
        self.assertGreater(core_score, alias_score)
        self.assertEqual(scene_score, 0)

    def test_voice_cards_expose_semantic_attributes(self):
        voice_cards = [
            card
            for card in self.packet["cards"]
            if card["card_id"].startswith(("style-guofeng-", "style-rap-"))
        ]
        self.assertEqual(len(voice_cards), 7)
        for card in voice_cards:
            self.assertIn(card["vocal_type"], {"female_sung", "sung_any", "humming_or_none", "optional_humming", "optional_sung_or_instrumental", "breathy_rap", "half_spoken_rap", "warm_breathy_rap", "instrumental_or_short_vocal", "vocal_chops_or_sung"})
            self.assertIsInstance(card["rap_default"], bool)
            self.assertIsInstance(card["instrumental_default"], bool)
            self.assertEqual(card["default_language"], "zh-CN")


if __name__ == "__main__":
    unittest.main()
