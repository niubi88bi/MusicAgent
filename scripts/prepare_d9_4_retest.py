"""从已完成首轮标注中分层抽取盲化隔时复标子集。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from scripts.prepare_d9_4_annotation import SCHEMA_VERSION, annotation_template


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} 必须是 JSON 对象")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _max_label(row: dict[str, object], field: str) -> int:
    values = [
        int(line[field])
        for line in row.get("lines", [])
        if isinstance(line, dict) and line.get(field) is not None
    ]
    if not values:
        raise ValueError(f"样本 {row.get('sample_id')} 的 {field} 尚未完成")
    return max(values)


def select_retest_cards(
    cards: list[dict[str, object]],
    manifest: list[dict[str, object]],
    round1: list[dict[str, object]],
    *,
    seed: int,
    fraction: float = 0.30,
    minimum: int = 40,
) -> list[dict[str, object]]:
    """按来源、段落和两类最高标签分层，抽取不暴露首轮结果的卡片。"""

    if not 0 < fraction <= 1:
        raise ValueError("fraction 必须在 (0, 1] 内")
    cards_by_id = {str(row["sample_id"]): row for row in cards}
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    round1_by_id = {str(row["sample_id"]): row for row in round1}
    common_ids = sorted(set(cards_by_id) & set(manifest_by_id) & set(round1_by_id))
    if not common_ids:
        raise ValueError("卡片、私有清单和首轮标注没有共同 sample_id")

    target = min(len(common_ids), max(minimum, math.ceil(len(common_ids) * fraction)))
    strata: dict[tuple[str, str, int, int], list[str]] = defaultdict(list)
    for sample_id in common_ids:
        manifest_row = manifest_by_id[sample_id]
        annotation_row = round1_by_id[sample_id]
        key = (
            str(manifest_row.get("source_type", "unknown")),
            str(manifest_row.get("section", "unknown")),
            _max_label(annotation_row, "length_fit"),
            _max_label(annotation_row, "pause_fit"),
        )
        strata[key].append(sample_id)

    rng = random.Random(seed)
    buckets = list(strata.values())
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    selected: list[str] = []
    while len(selected) < target:
        made_progress = False
        for bucket in buckets:
            if bucket and len(selected) < target:
                selected.append(bucket.pop())
                made_progress = True
        if not made_progress:
            break
    rng.shuffle(selected)
    return [cards_by_id[sample_id] for sample_id in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--fraction", type=float, default=0.30)
    parser.add_argument("--minimum", type=int, default=40)
    args = parser.parse_args()

    cards = select_retest_cards(
        _read_jsonl(args.cards),
        _read_jsonl(args.manifest),
        _read_jsonl(args.round1),
        seed=args.seed,
        fraction=args.fraction,
        minimum=args.minimum,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    _write_jsonl(args.out / "cards.round2.jsonl", cards)
    _write_jsonl(
        args.out / "annotations.round2.jsonl",
        annotation_template(cards, "round2"),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "selection": "stratified_single_annotator_retest",
        "seed": args.seed,
        "fraction": args.fraction,
        "minimum": args.minimum,
        "selected_cards": len(cards),
    }
    (args.out / "retest_selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.out.resolve())


if __name__ == "__main__":
    main()
