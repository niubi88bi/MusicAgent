"""完整歌曲 Web 模式离线准入测试。"""

from pathlib import Path
import unittest

from agents.lyricist_agent import LyricistAgent
from test_d7_retry import GOOD_DRAFT, REQUIREMENT
from tools.lyrics_quality.full_song_audit import audit_full_song


class FullSongWebModeTests(unittest.TestCase):
    def test_full_lyricist_uses_long_prompt_without_changing_preview(self):
        class CaptureLLM:
            def __init__(self):
                self.calls = []

            def chat_json(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                return GOOD_DRAFT.model_dump()

        llm = CaptureLLM()
        LyricistAgent(llm, persona="structured").run(REQUIREMENT)
        LyricistAgent(llm, persona="structured", full_song=True).run(REQUIREMENT)
        self.assertEqual([call[1]["max_tokens"] for call in llm.calls], [2200, 3000])
        self.assertNotIn("至少 24 行", llm.calls[0][0][0]["content"])
        self.assertIn("至少 24 行", llm.calls[1][0][0]["content"])

    def test_accepted_long_lyrics_from_manual_validation_pass_text_gate(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "data/output/full-song-120-validation/80953bde339845c4997309b550f79c5b/lyrics.txt"
        if not path.is_file():
            self.skipTest("本地试听产物未保留")
        report = audit_full_song(path.read_text(encoding="utf-8"), "晚风陪我回家 吹散一天的疲乏")
        self.assertTrue(report["passed"])
        self.assertEqual(report["singing_line_count"], 28)

    def test_short_lyrics_fail_closed(self):
        report = audit_full_song("[Verse]\n晚风陪我回家\n[Chorus]\n晚风陪我回家\n[Bridge]\n回家", "晚风陪我回家")
        self.assertFalse(report["passed"])
        self.assertGreater(len(report["issues"]), 0)


if __name__ == "__main__":
    unittest.main()
