from __future__ import annotations

import unittest

from scripts.validate_full_song_120s import (
    RETEST_PRONUNCIATION_REPLACEMENT,
    RETEST_PRONUNCIATION_SOURCE,
    RETEST_TERMINAL_VERSE,
    build_manual_feedback_retest_lyrics,
)


class FullSong120RetestTests(unittest.TestCase):
    def _source_lyrics(self) -> str:
        chorus_line = f"路灯陪我回家 {RETEST_PRONUNCIATION_SOURCE}"
        return (
            "[Verse]\n第一段第一行\n第一段第二行\n\n"
            f"[Chorus]\n{chorus_line}\n\n"
            f"[Chorus]\n{chorus_line}\n\n"
            f"[Chorus]\n{chorus_line}\n\n"
            f"{RETEST_TERMINAL_VERSE}"
        )

    def test_applies_only_registered_replacements(self) -> None:
        source = self._source_lyrics()

        revised, report = build_manual_feedback_retest_lyrics(source)

        self.assertNotIn(RETEST_PRONUNCIATION_SOURCE, revised)
        self.assertEqual(revised.count(RETEST_PRONUNCIATION_REPLACEMENT), 3)
        self.assertNotIn(RETEST_TERMINAL_VERSE, revised)
        self.assertEqual(report["pronunciation_patch"]["occurrences"], 3)
        self.assertEqual(
            revised,
            source[: -len(RETEST_TERMINAL_VERSE)]
            .rstrip()
            .replace(
                RETEST_PRONUNCIATION_SOURCE,
                RETEST_PRONUNCIATION_REPLACEMENT,
            ),
        )

    def test_fails_closed_when_pronunciation_count_changes(self) -> None:
        source = self._source_lyrics().replace(RETEST_PRONUNCIATION_SOURCE, "别的词", 1)

        with self.assertRaisesRegex(ValueError, "发音修订片段数量"):
            build_manual_feedback_retest_lyrics(source)

    def test_fails_closed_when_terminal_verse_is_not_at_end(self) -> None:
        source = self._source_lyrics() + "\n额外一行"

        with self.assertRaisesRegex(ValueError, "末尾 Verse"):
            build_manual_feedback_retest_lyrics(source)


if __name__ == "__main__":
    unittest.main()
