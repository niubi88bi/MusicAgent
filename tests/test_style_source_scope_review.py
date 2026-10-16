import unittest
from pathlib import Path

from scripts.build_style_source_scope_review import (
    DEFAULT_ADMISSION_AUDIT_PATH,
    DEFAULT_CARDS_PATH,
    DEFAULT_SOURCES_PATH,
    build_review_packet,
    render_markdown,
)


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/miaoxiang/style_templates.v2.jsonl").is_file(), "private style-card dataset is not published")
class StyleSourceScopeReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = build_review_packet(
            DEFAULT_CARDS_PATH,
            DEFAULT_SOURCES_PATH,
            DEFAULT_ADMISSION_AUDIT_PATH,
        )

    def test_packet_covers_all_sources_and_cards_with_local_demo_decisions(self) -> None:
        self.assertEqual(self.packet["summary"]["source_count"], 7)
        self.assertEqual(self.packet["summary"]["card_count"], 17)
        self.assertEqual(self.packet["summary"]["source_decisions_pending"], 0)
        self.assertEqual(self.packet["summary"]["card_decisions_pending"], 0)
        self.assertTrue(
            all(row["decision"] == "local_demo_allowed" for row in self.packet["source_reviews"])
        )
        self.assertTrue(
            all(row["decision"] == "local_demo_allowed" for row in self.packet["card_reviews"])
        )

    def test_packet_admits_direct_cards_only_for_local_demo(self) -> None:
        self.assertEqual(self.packet["summary"]["local_demo_eligible_count"], 16)
        self.assertEqual(self.packet["summary"]["public_product_eligible_count"], 0)
        self.assertFalse(self.packet["summary"]["review_status_changed"])
        self.assertFalse(self.packet["summary"]["index_changed"])

    def test_wikipedia_records_keep_attribution_and_share_alike_notice(self) -> None:
        wikipedia = [
            row
            for row in self.packet["source_reviews"]
            if row["license_status"] == "CC-BY-SA-4.0"
        ]
        self.assertEqual(len(wikipedia), 2)
        for row in wikipedia:
            self.assertTrue(row["attribution_required"])
            self.assertTrue(row["share_alike_required"])
            self.assertEqual(len(row["supplemental_attribution_records"]), 1)
            detail = row["supplemental_attribution_records"][0]
            self.assertTrue(detail["attribution"])
            self.assertTrue(detail["changes"])

    def test_tencent_source_is_marked_as_design_reference_only(self) -> None:
        source = next(
            row
            for row in self.packet["source_reviews"]
            if row["source_id"] == "tencent-media-lab-smart-music-tags-20260914"
        )
        self.assertTrue(source["design_reference_only"])
        self.assertEqual(source["primary_card_ids"], [])
        self.assertEqual(source["supplemental_card_ids"], [])

    def test_markdown_records_local_demo_decisions(self) -> None:
        markdown = render_markdown(self.packet)
        self.assertIn("不构成法律意见", markdown)
        self.assertNotIn("待填写", markdown)
        self.assertEqual(markdown.count("local_demo_allowed"), 25)


if __name__ == "__main__":
    unittest.main()
