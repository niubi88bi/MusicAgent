"""Public, self-authored fixture for the style-card contract."""

import json
import unittest
from pathlib import Path

from scripts.validate_style_rag_data import validate_dataset
from tools.rag.schemas import SourceRecord, StyleCard, compute_style_card_hash
from tools.rag.style_index import build_embedding_document


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/public_demo"


class PublicStyleDemoTests(unittest.TestCase):
    def test_self_authored_card_parses_and_hashes(self) -> None:
        source = SourceRecord.model_validate(
            json.loads((DEMO / "source_catalog.jsonl").read_text(encoding="utf-8"))
        )
        card = StyleCard.model_validate(
            json.loads((DEMO / "style_cards.jsonl").read_text(encoding="utf-8"))
        )
        self.assertEqual(card.source_id, source.source_id)
        self.assertEqual(card.source_type, "project_original")
        self.assertEqual(card.official_facts, [])
        self.assertEqual(card.content_sha256, compute_style_card_hash(card))
        self.assertIn(card.genre, build_embedding_document(card))

    def test_demo_is_valid_but_not_admitted_to_any_index(self) -> None:
        report = validate_dataset(
            DEMO / "style_cards.jsonl",
            DEMO / "source_catalog.jsonl",
            DEMO / "admission_audit.json",
        )
        self.assertEqual(report["card_count"], 1)
        self.assertEqual(report["source_count"], 1)
        self.assertEqual(report["valid_card_count"], 1)
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["local_demo_eligible_count"], 0)
        self.assertEqual(report["public_product_eligible_count"], 0)


if __name__ == "__main__":
    unittest.main()
