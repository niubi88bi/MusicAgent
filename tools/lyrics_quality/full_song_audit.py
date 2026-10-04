"""完整歌曲的保守文本准入；不声称音频已唱完。"""

from collections import Counter
import re

from tools.lyrics_quality.parser import parse_lyric_lines


def audit_full_song(lyrics: str, hook: str) -> dict[str, object]:
    """只检查可确定的结构和长度，供长歌音频提交前失败关闭。"""
    blocks: list[tuple[str, list[str]]] = []
    for raw in lyrics.splitlines():
        line = raw.strip()
        match = re.fullmatch(r"\[([^\]]+)\]", line)
        if match:
            blocks.append((match.group(1).strip().lower(), []))
        elif line and blocks:
            blocks[-1][1].append(line)
    lines = [line for _, section in blocks for line in section if re.search(r"[\u4e00-\u9fff]", line)]
    counts = Counter(label for label, _ in blocks)
    chinese_chars = sum("\u4e00" <= char <= "\u9fff" for line in lines for char in line)
    parsed_counts = Counter(line.section for line in parse_lyric_lines(lyrics))
    hook_count = sum(1 for label, section in blocks if label == "chorus" for line in section if line == hook.strip())
    issues: list[str] = []
    if len(lines) < 24:
        issues.append(f"演唱行不足 24 行（当前 {len(lines)} 行）")
    if chinese_chars < 260:
        issues.append(f"中文歌词不足 260 字（当前 {chinese_chars} 字）")
    if counts["verse"] < 2 or parsed_counts["Verse"] < 8:
        issues.append("至少需要两段主歌、共 8 行")
    if counts["chorus"] < 2 or parsed_counts["Chorus"] < 8:
        issues.append("至少需要两段副歌、共 8 行")
    if any(len(section) < 4 for label, section in blocks if label in {"verse", "chorus"}):
        issues.append("每段主歌和副歌至少需要 4 行")
    if counts["bridge"] < 1:
        issues.append("缺少过渡段 Bridge")
    if any(not section for _, section in blocks):
        issues.append("存在空段落")
    if not hook.strip() or hook_count < 2:
        issues.append("Hook 未在至少两段副歌中完整出现")
    if any(marker in lyrics for marker in ("待续", "未完", "省略", "...", "……")):
        issues.append("歌词中存在未完成标记")
    return {
        "passed": not issues,
        "issues": issues,
        "singing_line_count": len(lines),
        "chinese_character_count": chinese_chars,
        "hook_occurrences_in_chorus": hook_count,
        "scope": "文本结构与长度准入；音频歌词覆盖和发音仍需人工试听",
    }
