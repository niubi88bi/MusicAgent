"""离线计算 D9.4 误报、漏报和隔时复标一致性，不改业务评分。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.lyrics_quality.d9_4_evaluation import (
    binary_metrics,
    intrarater_metrics,
    is_structural_marker,
    score_gate,
)


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


def _structural_line_ids(manifest: list[dict[str, object]]) -> set[str]:
    return {
        str(line["line_id"])
        for row in manifest
        for line in row.get("lines", [])
        if isinstance(line, dict) and is_structural_marker(line.get("text"))
    }


def _annotation_index(
    rows: list[dict[str, object]],
    field: str,
    *,
    excluded_line_ids: set[str] | None = None,
) -> dict[str, int]:
    result: dict[str, int] = {}
    excluded = excluded_line_ids or set()
    for row in rows:
        for line in row.get("lines", []):
            if not isinstance(line, dict) or line.get(field) is None:
                continue
            line_id = str(line["line_id"])
            if line_id not in excluded:
                result[line_id] = int(line[field])
    return result


def _prediction_index(
    rows: list[dict[str, object]], field: str, *, excluded_line_ids: set[str] | None = None
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    excluded = excluded_line_ids or set()
    for row in rows:
        for line in row.get("lines", []):
            if isinstance(line, dict):
                line_id = str(line["line_id"])
                if line_id not in excluded:
                    result[line_id] = bool(line[field])
    return result


def _evaluate_rule(
    predictions: dict[str, bool],
    labels: dict[str, int],
    *,
    minimum_positive_examples: int,
    minimum_negative_examples: int,
) -> dict[str, object]:
    common_ids = sorted(set(predictions) & set(labels))
    pairs = [(predictions[item], labels[item]) for item in common_ids]
    metrics = binary_metrics(pairs)
    return {
        "evaluated_items": len(common_ids),
        "metrics": metrics,
        "borderline_as_negative": binary_metrics(pairs, borderline="negative"),
        "borderline_as_positive": binary_metrics(pairs, borderline="positive"),
        "detection_gate_only": score_gate(
            metrics,
            minimum_positive_examples=minimum_positive_examples,
            minimum_negative_examples=minimum_negative_examples,
        ),
    }


def _evaluate_rule_by_split(
    manifest: list[dict[str, object]],
    prediction_field: str,
    labels: dict[str, int],
    excluded_line_ids: set[str],
) -> dict[str, object]:
    result = {
        "all": _evaluate_rule(
            _prediction_index(manifest, prediction_field, excluded_line_ids=excluded_line_ids),
            labels,
            minimum_positive_examples=50,
            minimum_negative_examples=100,
        )
    }
    for split in ("development", "validation"):
        split_rows = [row for row in manifest if row.get("dataset_split") == split]
        result[split] = _evaluate_rule(
            _prediction_index(split_rows, prediction_field, excluded_line_ids=excluded_line_ids),
            labels,
            minimum_positive_examples=30 if split == "development" else 20,
            minimum_negative_examples=60 if split == "development" else 40,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--round2", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = _read_jsonl(args.manifest)
    round1 = _read_jsonl(args.round1)
    excluded_line_ids = _structural_line_ids(manifest)
    length_labels = _annotation_index(
        round1, "length_fit", excluded_line_ids=excluded_line_ids
    )
    pause_labels = _annotation_index(
        round1, "pause_fit", excluded_line_ids=excluded_line_ids
    )
    report: dict[str, object] = {
        "method": "d9_4_single_annotator_v1",
        "claim_boundary": "仅用于 D9.4 内部准入，不是客观准确率或普适可唱性。",
        "ready_to_change_business_score": False,
        "score_change_blocker": "还需复标一致性、子组检查和至少 24 组盲化修改前后证据。",
        "length": _evaluate_rule_by_split(
            manifest, "predicted_length_issue", length_labels, excluded_line_ids
        ),
        "pause": _evaluate_rule_by_split(
            manifest, "predicted_pause_issue", pause_labels, excluded_line_ids
        ),
    }
    if args.round2:
        round2 = _read_jsonl(args.round2)
        report["intrarater"] = {
            "length": intrarater_metrics(
                length_labels,
                _annotation_index(
                    round2, "length_fit", excluded_line_ids=excluded_line_ids
                ),
            ),
            "pause": intrarater_metrics(
                pause_labels,
                _annotation_index(
                    round2, "pause_fit", excluded_line_ids=excluded_line_ids
                ),
            ),
        }
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.out.resolve())


if __name__ == "__main__":
    main()
