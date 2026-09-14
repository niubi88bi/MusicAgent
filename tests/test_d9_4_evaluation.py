"""D9.4 标注指标与盲化种子集的离线测试。"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_d9_4_annotation import (
    _make_length_variant,
    _make_pause_variant,
    annotation_template,
    build_rows,
)
from scripts.prepare_d9_4_retest import select_retest_cards
from scripts.evaluate_d9_4_annotations import _annotation_index, _prediction_index, _structural_line_ids
from tools.lyrics_quality.d9_4_evaluation import (
    binary_metrics,
    intrarater_metrics,
    is_structural_marker,
    score_gate,
    weighted_kappa,
    wilson_interval,
)


class D94EvaluationTests(unittest.TestCase):
    def test_binary_metrics_exclude_borderline_labels(self) -> None:
        metrics = binary_metrics([(True, 2), (True, 0), (False, 2), (False, 0), (True, 1)])

        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (1, 1, 1, 1))
        self.assertEqual(metrics["borderline"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["false_negative_rate"], 0.5)
        strict = binary_metrics([(True, 1)], borderline="negative")
        liberal = binary_metrics([(False, 1)], borderline="positive")
        self.assertEqual(strict["fp"], 1)
        self.assertEqual(liberal["fn"], 1)

    def test_wilson_interval_is_bounded(self) -> None:
        interval = wilson_interval(8, 10)

        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertLess(interval[0], 0.8)
        self.assertGreater(interval[1], 0.8)
        self.assertGreaterEqual(interval[0], 0.0)
        self.assertLessEqual(interval[1], 1.0)

    def test_weighted_kappa_and_intrarater_metrics(self) -> None:
        first = {"a": 0, "b": 1, "c": 2, "d": 0}
        second = {"a": 0, "b": 1, "c": 2, "d": 2}

        metrics = intrarater_metrics(first, second)

        self.assertAlmostEqual(metrics["exact_agreement"], 0.75)
        self.assertAlmostEqual(metrics["binary_hard_issue_agreement"], 0.75)
        self.assertAlmostEqual(metrics["severe_flip_rate"], 0.25)
        self.assertEqual(weighted_kappa([0, 1, 2], [0, 1, 2]), 1.0)

    def test_score_gate_requires_counts_and_quality(self) -> None:
        metrics = binary_metrics(
            [(True, 2)] * 18
            + [(False, 2)] * 2
            + [(True, 0)] * 4
            + [(False, 0)] * 36
        )

        gate = score_gate(metrics)

        self.assertTrue(gate["eligible"])

    def test_seed_cards_hide_predictions_and_source_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir="tmp") as temporary_dir:
            root = Path(temporary_dir)
            case_dir = root / "case"
            case_dir.mkdir()
            (case_dir / "selected_lyrics.txt").write_text(
                "[Verse]\n晚风吹过街角\n灯火照亮归途\n\n"
                "[Chorus]\n我们一起回家\n我们不再害怕\n\n"
                "[Bridge]\n明天仍会出发\n此刻先拥抱吧",
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {"cases": [{"case_id": "case", "source_dir": str(case_dir)}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            cards, private_rows = build_rows(summary, seed=7)
            templates = annotation_template(cards, "round1")

            self.assertEqual(len(cards), 3)
            self.assertEqual(len(private_rows), 3)
            self.assertEqual(len(templates), 3)
            self.assertRegex(str(cards[0]["sample_id"]), r"^D94-S\d{4}$")
            self.assertNotIn("source_path", cards[0])
            self.assertNotIn("source_sample_key", cards[0])
            self.assertNotIn("predicted_length_issue", cards[0]["lines"][0])
            self.assertIsNone(templates[0]["lines"][0]["length_fit"])

    def test_retest_selection_is_blind_and_repeatable(self) -> None:
        cards = [
            {
                "sample_id": f"s{index}",
                "section": "Verse" if index % 2 else "Chorus",
                "lines": [{"line_id": f"s{index}-l1", "text": "测试歌词"}],
            }
            for index in range(10)
        ]
        manifest = [
            {
                "sample_id": f"s{index}",
                "source_type": "natural" if index < 5 else "controlled_length",
                "section": cards[index]["section"],
            }
            for index in range(10)
        ]
        round1 = [
            {
                "sample_id": f"s{index}",
                "lines": [
                    {
                        "line_id": f"s{index}-l1",
                        "length_fit": index % 3,
                        "pause_fit": (index + 1) % 3,
                    }
                ],
            }
            for index in range(10)
        ]

        selected = select_retest_cards(
            cards, manifest, round1, seed=11, fraction=0.30, minimum=4
        )
        repeated = select_retest_cards(
            cards, manifest, round1, seed=11, fraction=0.30, minimum=4
        )

        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 4)
        self.assertNotIn("source_type", selected[0])
        self.assertNotIn("length_fit", selected[0]["lines"][0])

    def test_controlled_variants_keep_family_and_change_two_lines(self) -> None:
        row = {
            "source_sample_key": "family--structured--verse-1",
            "family_id": "family",
            "variant_id": "structured",
            "source_type": "natural_generated",
            "source_path": "fixture",
            "dataset_split": "development",
            "section": "Verse",
            "stanza_index": 1,
            "lines": [
                {"line_position": 1, "source_line_no": 2, "text": "晚风吹过安静街角"},
                {"line_position": 2, "source_line_no": 3, "text": "灯火照亮回家小路"},
                {"line_position": 3, "source_line_no": 4, "text": "脚步慢慢走向远方"},
                {"line_position": 4, "source_line_no": 5, "text": "月光落在我的肩上"},
            ],
        }

        length_variant = _make_length_variant(row, random.Random(3))
        pause_variant = _make_pause_variant(row)

        self.assertEqual(length_variant["family_id"], "family")
        self.assertEqual(pause_variant["family_id"], "family")
        self.assertEqual(len(length_variant["mutation"]["target_positions"]), 2)
        self.assertEqual(len(pause_variant["mutation"]["target_positions"]), 2)
        self.assertEqual(length_variant["source_type"], "controlled_length")
        self.assertEqual(pause_variant["source_type"], "controlled_pause")

    def test_structural_markers_are_excluded_from_annotation_metrics(self) -> None:
        manifest = [{
            "lines": [
                {"line_id": "marker", "text": "[Outro]", "predicted_length_issue": True},
                {"line_id": "lyric", "text": "晚风吹过街角", "predicted_length_issue": False},
            ]
        }]
        annotations = [{
            "lines": [
                {"line_id": "marker", "length_fit": 2},
                {"line_id": "lyric", "length_fit": 0},
            ]
        }]

        excluded = _structural_line_ids(manifest)
        labels = _annotation_index(annotations, "length_fit", excluded_line_ids=excluded)
        predictions = _prediction_index(
            manifest, "predicted_length_issue", excluded_line_ids=excluded
        )

        self.assertTrue(is_structural_marker("[Outro]"))
        self.assertNotIn("marker", labels)
        self.assertNotIn("marker", predictions)
        self.assertEqual(labels, {"lyric": 0})


if __name__ == "__main__":
    unittest.main()
