"""从 D9 影子评估案例生成盲化段落卡、私有清单和首轮标注模板。"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from tools.lyrics_quality.parser import parse_lyric_lines
from tools.lyrics_quality.d9_4_evaluation import is_structural_marker
from tools.lyrics_quality.rhythm_checker import check_rhythm


SCHEMA_VERSION = "d9.4-v1"
CHINESE_ONLY = re.compile(r"[\u4e00-\u9fff]")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_lyrics(case_dir: Path) -> str:
    for name in ("selected_lyrics.txt", "lyrics.txt"):
        path = case_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    draft_path = case_dir / "lyric_draft.json"
    if draft_path.exists():
        payload = _read_json(draft_path)
        if isinstance(payload, dict) and isinstance(payload.get("lyrics"), str):
            return payload["lyrics"].strip()
    raise FileNotFoundError(f"{case_dir} 缺少可读取的歌词文件")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_cases(summary_path: Path) -> list[dict[str, object]]:
    payload = _read_json(summary_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("影子评估摘要缺少 cases 列表")
    return [case for case in payload["cases"] if isinstance(case, dict)]


def _append_lyrics_rows(
    private_rows: list[dict[str, object]],
    *,
    family_id: str,
    variant_id: str,
    source_type: str,
    source_path: Path,
    lyrics: str,
    seen_stanzas: set[str],
) -> None:
    """把一份歌词拆为去重段落，并保存当前检测预测。"""

    parsed = parse_lyric_lines(lyrics)
    rhythm = check_rhythm(lyrics)
    analyses = {item.line_no: item for item in rhythm.lines}
    issues_by_line: dict[int, set[str]] = defaultdict(set)
    for issue in rhythm.issues:
        if issue.line_no is not None:
            issues_by_line[issue.line_no].add(issue.code)
    grouped: dict[tuple[str, int], list[object]] = defaultdict(list)
    for line in parsed:
        grouped[(line.section, line.stanza_index)].append(line)

    for (section, stanza_index), lines in grouped.items():
        normalized_stanza = "".join(
            CHINESE_ONLY.findall("".join(line.text for line in lines))
        )
        if not normalized_stanza or normalized_stanza in seen_stanzas:
            continue
        seen_stanzas.add(normalized_stanza)
        source_sample_key = f"{family_id}--{variant_id}--{section.lower()}-{stanza_index}"
        line_rows: list[dict[str, object]] = []
        for position, line in enumerate(lines, start=1):
            if is_structural_marker(line.text):
                continue
            analysis = analyses[line.line_no]
            codes = issues_by_line[line.line_no]
            line_rows.append(
                {
                    "line_position": position,
                    "source_line_no": line.line_no,
                    "text": line.text,
                    "chinese_char_count": analysis.chinese_char_count,
                    "phrase_count": analysis.phrase_count,
                    "predicted_length_issue": "line_length_outlier" in codes,
                    "predicted_pause_issue": "phrase_count_mismatch" in codes,
                }
            )
        if not line_rows:
            continue
        private_rows.append(
            {
                "source_sample_key": source_sample_key,
                "family_id": family_id,
                "variant_id": variant_id,
                "source_type": source_type,
                "source_path": str(source_path.resolve()),
                "section": section,
                "stanza_index": stanza_index,
                "lines": line_rows,
            }
        )


def _load_generated_pool(pool_dir: Path) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for candidates_path in sorted(pool_dir.glob("*/lyric_candidates.json")):
        family_id = candidates_path.parent.name
        payload = _read_json(candidates_path)
        if not isinstance(payload, list):
            continue
        for candidate in payload:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("draft"), dict):
                continue
            draft = candidate["draft"]
            lyrics = draft.get("lyrics")
            if isinstance(lyrics, str):
                sources.append(
                    {
                        "family_id": family_id,
                        "variant_id": str(candidate.get("persona_id", "candidate")),
                        "source_path": candidates_path,
                        "lyrics": lyrics,
                    }
                )
    return sources


def _balanced_natural_selection(
    rows: list[dict[str, object]], limit: int | None, rng: random.Random
) -> list[dict[str, object]]:
    if limit is None or len(rows) <= limit:
        return rows
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"])].append(row)
    buckets = list(grouped.values())
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    selected: list[dict[str, object]] = []
    while len(selected) < limit:
        made_progress = False
        for bucket in buckets:
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def _assign_family_splits(
    rows: list[dict[str, object]],
    *,
    forced_development: set[str],
    rng: random.Random,
) -> None:
    families = sorted({str(row["family_id"]) for row in rows})
    validation_target = max(1, round(len(families) / 3))
    eligible = [family for family in families if family not in forced_development]
    if len(eligible) < validation_target:
        raise ValueError("没有足够的新歌词家族建立锁定验证集")
    validation = set(rng.sample(eligible, validation_target))
    for row in rows:
        row["dataset_split"] = (
            "validation" if str(row["family_id"]) in validation else "development"
        )


def _select_control_sources(
    rows: list[dict[str, object]], count: int, rng: random.Random
) -> list[dict[str, object]]:
    """让约三分之一受控卡来自锁定验证家族。"""

    if count <= 0:
        return []
    development = [row for row in rows if row["dataset_split"] == "development"]
    validation = [row for row in rows if row["dataset_split"] == "validation"]
    rng.shuffle(development)
    rng.shuffle(validation)
    validation_count = min(len(validation), round(count / 3))
    development_count = min(len(development), count - validation_count)
    selected = [
        *validation[:validation_count],
        *development[:development_count],
    ]
    if len(selected) < count:
        used = {str(row["source_sample_key"]) for row in selected}
        remainder = [
            row for row in rows if str(row["source_sample_key"]) not in used
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    if len(selected) < count:
        raise ValueError("自然段落不足，无法构造指定数量的受控变体")
    rng.shuffle(selected)
    return selected


def _recompute_mutated_row(row: dict[str, object], texts: list[str]) -> dict[str, object]:
    mutated = copy.deepcopy(row)
    section = str(row["section"])
    report = check_rhythm(f"[{section}]\n" + "\n".join(texts))
    issues_by_position: dict[int, set[str]] = defaultdict(set)
    source_lines = parse_lyric_lines(f"[{section}]\n" + "\n".join(texts))
    position_by_line_no = {line.line_no: position for position, line in enumerate(source_lines, start=1)}
    for issue in report.issues:
        if issue.line_no in position_by_line_no:
            issues_by_position[position_by_line_no[issue.line_no]].add(issue.code)
    analyses = {position: item for position, item in enumerate(report.lines, start=1)}
    for position, line in enumerate(mutated["lines"], start=1):
        assert isinstance(line, dict)
        analysis = analyses[position]
        line["text"] = texts[position - 1]
        line["chinese_char_count"] = analysis.chinese_char_count
        line["phrase_count"] = analysis.phrase_count
        line["predicted_length_issue"] = "line_length_outlier" in issues_by_position[position]
        line["predicted_pause_issue"] = "phrase_count_mismatch" in issues_by_position[position]
    return mutated


def _make_length_variant(row: dict[str, object], rng: random.Random) -> dict[str, object]:
    lines = row["lines"]
    assert isinstance(lines, list)
    texts = [str(line["text"]) for line in lines if isinstance(line, dict)]
    targets = rng.sample(range(len(texts)), k=min(2, len(texts)))
    extensions = (
        "仍沿着漫长街巷反复寻找那些没有说完的旧日回忆",
        "却又穿过很远很远的夜色追问许多年前没有答案的往事",
    )
    for target, extension in zip(targets, extensions):
        characters = "".join(CHINESE_ONLY.findall(texts[target]))
        ending = characters[-1:] or "啊"
        stem = characters[:-1]
        texts[target] = stem + extension + ending
    mutated = _recompute_mutated_row(row, texts)
    mutated["source_type"] = "controlled_length"
    mutated["variant_id"] = f"{row['variant_id']}-length"
    mutated["source_sample_key"] = f"{row['source_sample_key']}--length"
    mutated["mutation"] = {
        "kind": "length_only",
        "target_positions": [target + 1 for target in targets],
    }
    return mutated


def _make_pause_variant(row: dict[str, object]) -> dict[str, object]:
    lines = row["lines"]
    assert isinstance(lines, list)
    texts = [str(line["text"]) for line in lines if isinstance(line, dict)]
    targets = sorted(
        range(len(texts)),
        key=lambda index: len(CHINESE_ONLY.findall(texts[index])),
        reverse=True,
    )[: min(2, len(texts))]
    for offset, target in enumerate(targets):
        characters = "".join(CHINESE_ONLY.findall(texts[target]))
        desired_chunks = 5 + offset
        chunk_size = max(1, len(characters) // desired_chunks)
        chunks = [
            characters[index : index + chunk_size]
            for index in range(0, len(characters), chunk_size)
        ]
        texts[target] = "，".join(chunks)
    mutated = _recompute_mutated_row(row, texts)
    mutated["source_type"] = "controlled_pause"
    mutated["variant_id"] = f"{row['variant_id']}-pause"
    mutated["source_sample_key"] = f"{row['source_sample_key']}--pause"
    mutated["mutation"] = {
        "kind": "pause_only",
        "target_positions": [target + 1 for target in targets],
    }
    return mutated


def build_rows(
    summary_path: Path,
    *,
    seed: int,
    source_pool: Path | None = None,
    natural_limit: int | None = None,
    controlled_length_cards: int = 0,
    controlled_pause_cards: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """生成盲化卡片和带来源、预测的私有清单。"""

    natural_rows: list[dict[str, object]] = []
    forced_development: set[str] = set()
    seen_stanzas: set[str] = set()
    for case in _load_cases(summary_path):
        case_id = str(case["case_id"])
        forced_development.add(case_id)
        case_dir = Path(str(case["source_dir"]))
        lyrics = _read_lyrics(case_dir)
        _append_lyrics_rows(
            natural_rows,
            family_id=case_id,
            variant_id="selected",
            source_type="natural_saved_output",
            source_path=case_dir,
            lyrics=lyrics,
            seen_stanzas=seen_stanzas,
        )
    if source_pool is not None:
        for source in _load_generated_pool(source_pool):
            _append_lyrics_rows(
                natural_rows,
                family_id=str(source["family_id"]),
                variant_id=str(source["variant_id"]),
                source_type="natural_generated",
                source_path=Path(str(source["source_path"])),
                lyrics=str(source["lyrics"]),
                seen_stanzas=seen_stanzas,
            )

    rng = random.Random(seed)
    natural_rows = _balanced_natural_selection(natural_rows, natural_limit, rng)
    if source_pool is None:
        for row in natural_rows:
            row["dataset_split"] = "development"
    else:
        _assign_family_splits(
            natural_rows,
            forced_development=forced_development,
            rng=rng,
        )
    length_sources = _select_control_sources(natural_rows, controlled_length_cards, rng)
    pause_sources = _select_control_sources(natural_rows, controlled_pause_cards, rng)
    controlled_rows = [
        _make_length_variant(row, rng)
        for row in length_sources
    ]
    controlled_rows.extend(
        _make_pause_variant(row)
        for row in pause_sources
    )
    private_rows = [*natural_rows, *controlled_rows]
    rng.shuffle(private_rows)
    for index, row in enumerate(private_rows, start=1):
        sample_id = f"D94-S{index:04d}"
        row["sample_id"] = sample_id
        lines = row["lines"]
        assert isinstance(lines, list)
        for line in lines:
            assert isinstance(line, dict)
            line["line_id"] = f"{sample_id}-L{int(line['line_position']):02d}"
    cards = [
        {
            "sample_id": row["sample_id"],
            "section": row["section"],
            "lines": [
                {"line_id": line["line_id"], "text": line["text"]}
                for line in row["lines"]
            ],
        }
        for row in private_rows
    ]
    return cards, private_rows


def annotation_template(cards: Iterable[dict[str, object]], round_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for card in cards:
        lines = card["lines"]
        assert isinstance(lines, list)
        rows.append(
            {
                "sample_id": card["sample_id"],
                "annotator_id": "owner",
                "round_id": round_id,
                "label_schema_version": SCHEMA_VERSION,
                "annotated_at": None,
                "stanza_pattern": None,
                "rhythm_rating": None,
                "confidence": None,
                "needs_melody": None,
                "notes": "",
                "lines": [
                    {
                        "line_id": line["line_id"],
                        "length_fit": None,
                        "pause_fit": None,
                        "reason_code": None,
                        "notes": "",
                    }
                    for line in lines
                    if isinstance(line, dict)
                ],
            }
        )
    return rows


def coverage_report(private_rows: list[dict[str, object]]) -> dict[str, object]:
    sections = Counter(str(row["section"]) for row in private_rows)
    line_count = sum(len(row["lines"]) for row in private_rows if isinstance(row["lines"], list))
    families = {str(row["family_id"]) for row in private_rows}
    source_types = Counter(str(row["source_type"]) for row in private_rows)
    split_cards = Counter(str(row["dataset_split"]) for row in private_rows)
    split_families = {
        split: len(
            {
                str(row["family_id"])
                for row in private_rows
                if row["dataset_split"] == split
            }
        )
        for split in ("development", "validation")
    }
    target_counts = {
        "natural": 60,
        "controlled_length": 30,
        "controlled_pause": 30,
    }
    natural_count = source_types["natural_saved_output"] + source_types["natural_generated"]
    ready = (
        len(families) >= 16
        and len(private_rows) >= 120
        and natural_count >= target_counts["natural"]
        and source_types["controlled_length"] >= target_counts["controlled_length"]
        and source_types["controlled_pause"] >= target_counts["controlled_pause"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "current_seed": {
            "families": len(families),
            "stanza_cards": len(private_rows),
            "line_items": line_count,
            "sections": dict(sections),
            "source_types": dict(source_types),
            "split_cards": dict(split_cards),
            "split_families": split_families,
        },
        "target": {
            "families": 16,
            "stanza_cards": 120,
            "natural_cards": 60,
            "controlled_length_cards": 30,
            "controlled_pause_cards": 30,
        },
        "ready_for_annotation": ready,
        "ready_for_threshold_tuning": False,
        "reason": (
            "样本结构已达到首轮标注目标；仍需人工标签后才能调阈值。"
            if ready
            else "当前样本结构尚未达到首轮标注目标，且尚无人工标签。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--source-pool", type=Path)
    parser.add_argument("--natural-cards", type=int)
    parser.add_argument("--controlled-length-cards", type=int, default=0)
    parser.add_argument("--controlled-pause-cards", type=int, default=0)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="允许覆盖目标目录中的同名可再生产物，不删除其他文件。",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=args.replace)
    cards, private_rows = build_rows(
        args.shadow_summary,
        seed=args.seed,
        source_pool=args.source_pool,
        natural_limit=args.natural_cards,
        controlled_length_cards=args.controlled_length_cards,
        controlled_pause_cards=args.controlled_pause_cards,
    )
    _write_jsonl(args.out / "cards.round1.jsonl", cards)
    _write_jsonl(args.out / "manifest.private.jsonl", private_rows)
    _write_jsonl(
        args.out / "annotations.round1.jsonl",
        annotation_template(cards, "round1"),
    )
    report = coverage_report(private_rows)
    (args.out / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "README.md").write_text(
        "\n".join(
            [
                "# D9.4 首轮盲标包",
                "",
                "1. 启动 MusicAgent Web 服务并打开 `http://127.0.0.1:8000/d9-4-annotator.html`。",
                "2. 页面只载入 `cards.round1.jsonl` 和 `annotations.round1.jsonl`。",
                "3. 不要在首轮标注前打开 `manifest.private.jsonl`。",
                "4. 可随时导出部分进度；全部完成后再运行指标脚本。",
                "",
                f"当前段落卡：{report['current_seed']['stanza_cards']}；",
                f"逐行标注项：{report['current_seed']['line_items']}；",
                f"可开始首轮标注：{report['ready_for_annotation']}。",
            ]
        ),
        encoding="utf-8",
    )
    print(args.out.resolve())


if __name__ == "__main__":
    main()
