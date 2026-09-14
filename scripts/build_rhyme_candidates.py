"""Build a deterministic, human-review-only rhyme-word candidate list.

This script does not create the approved RAG lexicon. It reads the pinned Jieba
dictionary snapshot, uses the project's existing pypinyin/thirteen-rhyme logic,
and writes balanced candidates for later manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pypinyin import Style, lazy_pinyin, pinyin

from tools.lyrics_quality.thirteen_rhymes import RHYME_GROUPS, normalize_final, rhyme_group_for_final


DEFAULT_SOURCE_REVISION = "67fa2e36e72f69d9134b8a1037b83fbb070b9775"
DEFAULT_SOURCE_URL = "https://github.com/fxsjy/jieba/blob/master/jieba/dict.txt"
HAN_WORD_RE = re.compile(r"^[\u4e00-\u9fff]{1,2}$")

# Keep content words and a small set of idiom/location/time categories. Proper
# names, organizations, particles, conjunctions, numerals and pronouns are not
# useful default lyric endings and are left out of the first review batch.
ALLOWED_POS = {
    "a",
    "ad",
    "an",
    "d",
    "f",
    "i",
    "l",
    "n",
    "ng",
    "s",
    "t",
    "v",
    "vd",
    "vn",
}

STOP_WORDS = {
    "一个",
    "一些",
    "一样",
    "一般",
    "一切",
    "一旦",
    "一直",
    "不会",
    "不能",
    "不是",
    "不过",
    "为了",
    "以及",
    "任何",
    "作为",
    "其中",
    "其它",
    "其他",
    "关于",
    "具有",
    "可以",
    "同时",
    "因此",
    "如果",
    "已经",
    "应该",
    "当然",
    "很多",
    "所有",
    "所谓",
    "按照",
    "整个",
    "是否",
    "有关",
    "本身",
    "正在",
    "此外",
    "没有",
    "然后",
    "由于",
    "目前",
    "相关",
    "这种",
    "这些",
    "这样",
    "那个",
    "那些",
    "那么",
    "通过",
    "随着",
    "需要",
    "非常",
}


@dataclass(frozen=True)
class Candidate:
    word: str
    source_frequency: int
    source_pos: str
    pinyin_with_tone: str
    final: str
    rhyme_group: str
    tone_pattern: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("tmp/rag-source-research/jieba-raw/dict.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rhyme_lexicon.candidates.v1.jsonl"),
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("data/rhyme_candidate_seeds.v1.json"),
        help="Project-curated seed list to validate and enrich.",
    )
    parser.add_argument("--per-group", type=int, default=30)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tone_pattern(pinyin_value: str) -> str:
    match = re.search(r"([1-5])$", pinyin_value)
    if not match:
        return "待复核"
    tone = int(match.group(1))
    if tone in (1, 2):
        return "平"
    if tone in (3, 4):
        return "仄"
    return "轻声"


def has_ambiguous_last_character(word: str) -> bool:
    readings = {
        item
        for group in pinyin(word[-1], style=Style.TONE3, heteronym=True)
        for item in group
    }
    return len(readings) > 1


def candidate_from_row(word: str, frequency: int, source_pos: str) -> Candidate | None:
    if not HAN_WORD_RE.fullmatch(word):
        return None
    if word in STOP_WORDS or source_pos not in ALLOWED_POS:
        return None
    # The initial list deliberately excludes ambiguous ending characters. They
    # can be added later as context-specific records after manual pronunciation review.
    if has_ambiguous_last_character(word):
        return None

    word_pinyin = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
    word_finals = lazy_pinyin(word, style=Style.FINALS, strict=True, v_to_u=True)
    if not word_pinyin or not word_finals:
        return None
    final = normalize_final(word_finals[-1])
    rhyme_group = rhyme_group_for_final(final)
    if rhyme_group is None:
        return None
    return Candidate(
        word=word,
        source_frequency=frequency,
        source_pos=source_pos,
        pinyin_with_tone=" ".join(word_pinyin),
        final=final,
        rhyme_group=rhyme_group,
        tone_pattern=tone_pattern(word_pinyin[-1]),
    )


def read_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split()
            if len(parts) != 3:
                continue
            word, frequency_text, source_pos = parts
            try:
                frequency = int(frequency_text)
            except ValueError:
                continue
            candidate = candidate_from_row(word, frequency, source_pos)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def read_dictionary_index(path: Path) -> dict[str, tuple[int, str]]:
    index: dict[str, tuple[int, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split()
            if len(parts) != 3:
                continue
            word, frequency_text, source_pos = parts
            try:
                index[word] = (int(frequency_text), source_pos)
            except ValueError:
                continue
    return index


def read_seed_candidates(
    seed_path: Path,
    dictionary_index: dict[str, tuple[int, str]],
    per_group: int,
) -> list[Candidate]:
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"seed file has no groups object: {seed_path}")
    if set(groups) != set(RHYME_GROUPS):
        missing = sorted(set(RHYME_GROUPS) - set(groups))
        extra = sorted(set(groups) - set(RHYME_GROUPS))
        raise ValueError(f"seed rhyme groups differ from code; missing={missing}, extra={extra}")

    seen_words: set[str] = set()
    selected: list[Candidate] = []
    for expected_group in RHYME_GROUPS:
        words = groups[expected_group]
        if not isinstance(words, list) or len(words) != per_group:
            raise ValueError(
                f"seed group {expected_group!r} must contain exactly {per_group} words"
            )
        for word in words:
            if not isinstance(word, str) or not HAN_WORD_RE.fullmatch(word):
                raise ValueError(f"invalid seed word in {expected_group!r}: {word!r}")
            if word in seen_words:
                raise ValueError(f"duplicate seed word: {word}")
            seen_words.add(word)
            if word not in dictionary_index:
                raise ValueError(f"seed word is absent from pinned Jieba dictionary: {word}")
            source_frequency, source_pos = dictionary_index[word]
            word_pinyin = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
            word_finals = lazy_pinyin(word, style=Style.FINALS, strict=True, v_to_u=True)
            if not word_pinyin or not word_finals:
                raise ValueError(f"pypinyin could not analyse seed word: {word}")
            final = normalize_final(word_finals[-1])
            actual_group = rhyme_group_for_final(final)
            if actual_group != expected_group:
                raise ValueError(
                    f"seed word {word!r} expected {expected_group!r}, got "
                    f"final={final!r}, group={actual_group!r}"
                )
            selected.append(
                Candidate(
                    word=word,
                    source_frequency=source_frequency,
                    source_pos=source_pos,
                    pinyin_with_tone=" ".join(word_pinyin),
                    final=final,
                    rhyme_group=actual_group,
                    tone_pattern=tone_pattern(word_pinyin[-1]),
                )
            )
    return selected


def select_balanced(candidates: list[Candidate], per_group: int) -> list[Candidate]:
    if per_group <= 0:
        raise ValueError("per_group must be positive")

    by_group: dict[str, list[Candidate]] = {group: [] for group in RHYME_GROUPS}
    for candidate in candidates:
        by_group[candidate.rhyme_group].append(candidate)

    selected: list[Candidate] = []
    for group in RHYME_GROUPS:
        ranked = sorted(
            by_group[group],
            # Two-character content words are normally more useful than isolated
            # characters as retrieval suggestions; source frequency only ranks
            # candidates within the same source and is not treated as lyric frequency.
            key=lambda item: (-len(item.word), -item.source_frequency, item.word),
        )
        if len(ranked) < per_group:
            raise RuntimeError(
                f"rhyme group {group!r} has only {len(ranked)} eligible candidates; "
                f"requested {per_group}"
            )
        selected.extend(ranked[:per_group])
    return selected


def to_record(candidate: Candidate, source_hash: str, seed_path: Path) -> dict[str, object]:
    word_id = hashlib.sha1(candidate.word.encode("utf-8")).hexdigest()[:10]
    return {
        "id": f"rhyme-{candidate.rhyme_group}-{word_id}-v1",
        "schema_version": "1.0-candidate",
        "kind": "rhyme_word_candidate",
        "word": candidate.word,
        "pinyin": candidate.pinyin_with_tone,
        "final": candidate.final,
        "rhyme_group": candidate.rhyme_group,
        "tone_pattern": candidate.tone_pattern,
        "themes": [],
        "moods": [],
        "part_of_speech": candidate.source_pos,
        "frequency_tier": "pending_human_review",
        "source_type": "project_curated_jieba_verified_candidate",
        "source_title": "MusicAgent 人工候选 + Jieba dict.txt 收录核验",
        "source_url": DEFAULT_SOURCE_URL,
        "source_revision": DEFAULT_SOURCE_REVISION,
        "source_file_sha256": source_hash,
        "source_frequency": candidate.source_frequency,
        "source_frequency_note": "仅用于同一 Jieba 词典内候选排序，不代表歌词语料频率",
        "license_or_usage_note": "Jieba 仓库标注 MIT；词语、拼音和辙部为事实字段",
        "seed_file": seed_path.as_posix(),
        "selection_rule": "项目按歌词句尾可用性整理；固定 Jieba 快照核验收录；pypinyin 与现有十三辙代码校验读音和分组",
        "review_status": "draft",
        "human_review": {
            "decision": None,
            "pronunciation_verified": False,
            "lyric_ending_suitability": None,
            "theme_labels": [],
            "mood_labels": [],
            "notes": "",
        },
        "version": 1,
    }


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Jieba dictionary not found: {args.source}")
    if not args.seed.is_file():
        raise FileNotFoundError(f"project seed list not found: {args.seed}")
    source_hash = file_sha256(args.source)
    dictionary_index = read_dictionary_index(args.source)
    selected = read_seed_candidates(args.seed, dictionary_index, args.per_group)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in selected:
            handle.write(
                json.dumps(
                    to_record(candidate, source_hash, args.seed),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    counts = {group: 0 for group in RHYME_GROUPS}
    for candidate in selected:
        counts[candidate.rhyme_group] += 1
    print(json.dumps({"output": str(args.output), "total": len(selected), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
