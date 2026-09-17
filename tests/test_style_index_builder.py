import unittest

from tools.rag.build_style_index import prepare_build_report
from tools.rag.style_index import (
    build_embedding_document,
    build_metadata,
    expand_composite_query,
    load_index_inputs,
    select_local_demo_cards,
)


class StyleIndexBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards, cls.sources, cls.policy = load_index_inputs()
        cls.direct, cls.composite, cls.blocked = select_local_demo_cards(
            cls.cards, cls.sources, cls.policy
        )

    def test_dry_run_selects_sixteen_direct_cards_and_one_template(self) -> None:
        report, direct, composite = prepare_build_report()
        self.assertEqual(report["card_count"], 17)
        self.assertEqual(report["direct_candidate_count"], 16)
        self.assertEqual(report["composite_template_count"], 1)
        self.assertEqual(report["blocked_non_composite_count"], 0)
        self.assertTrue(report["all_direct_cards_approved"])
        self.assertFalse(report["index_changed"])
        self.assertEqual(len(direct), 16)
        self.assertEqual(len(composite), 1)

    def test_embedding_document_excludes_generation_and_license_fields(self) -> None:
        card = next(row for row in self.direct if row.id == "style-edm-future-bass-v2")
        document = build_embedding_document(card)
        self.assertIn("Future Bass", document)
        self.assertIn("Supersaw", document)
        self.assertNotIn(card.prompt_template, document)
        self.assertNotIn(card.license_or_usage_note, document)
        self.assertNotIn(card.backend_tag_source, document)
        for tag in card.scene_tags:
            self.assertNotIn(tag, document)

    def test_metadata_is_local_demo_and_traceable(self) -> None:
        card = self.direct[0]
        metadata = build_metadata(card)
        self.assertEqual(metadata["review_status"], "approved")
        self.assertEqual(metadata["card_role"], "style_card")
        self.assertEqual(metadata["retrieval_scope"], "direct_candidate")
        self.assertEqual(metadata["source_scope"], "local_demo_allowed")
        self.assertEqual(metadata["primary_source_id"], card.source_id)
        self.assertEqual(metadata["content_sha256"], card.content_sha256)

    def test_named_composite_expands_but_is_not_a_direct_card(self) -> None:
        expanded, template_ids = expand_composite_query(
            "苦情民谣电音 烟嗓 电子鼓",
            self.composite,
        )
        self.assertEqual(template_ids, ["style-dynamic-sad-folk-pop-remix-v2"])
        self.assertIn("Sad Mandopop", expanded)
        self.assertNotIn(template_ids[0], {card.id for card in self.direct})


if __name__ == "__main__":
    unittest.main()
