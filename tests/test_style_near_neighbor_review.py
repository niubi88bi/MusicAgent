import json
import unittest
from pathlib import Path

from scripts.prepare_style_near_neighbor_review import prepare, _read_jsonl


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/miaoxiang/style_templates.v2.jsonl").is_file(), "private style-card dataset is not published")
class StyleNearNeighborReviewTests(unittest.TestCase):
    def test_packet_covers_all_cards_without_admission_mutation(self):
        root = Path(__file__).resolve().parents[1]
        cards = _read_jsonl(root / "data/miaoxiang/style_templates.v2.jsonl")
        result = json.loads(
            (root / "data/output/evaluations/style-embedding-v1.1/result.json").read_text(encoding="utf-8")
        )
        packet = prepare(cards, result)
        covered = {card["card_id"] for group in packet["groups"] for card in group["cards"]}
        self.assertEqual(covered, {card["id"] for card in cards})
        self.assertEqual(packet["group_count"], 4)
        self.assertFalse(packet["review_status_mutated"])
        self.assertTrue(all(group["human_decision"] is None for group in packet["groups"]))


if __name__ == "__main__":
    unittest.main()
