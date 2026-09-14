"""D9.4 单人标注、隔时复标与节奏告警离线评估工具。"""

from __future__ import annotations

from collections import Counter
from math import sqrt
import re
from typing import Iterable, Mapping


HARD_ISSUE_LABEL = 2
LABELS = (0, 1, 2)
STRUCTURAL_MARKER_PATTERN = re.compile(
    r"^\[(?:Intro|Verse|Pre-?Chorus|Chorus|Bridge|Hook|Outro|Interlude|Instrumental)(?:\s+\d+)?\]$",
    re.IGNORECASE,
)


def is_structural_marker(text: object) -> bool:
    """判断歌词中的章节标签；标签不属于需要人工判断的歌词行。"""

    return isinstance(text, str) and bool(STRUCTURAL_MARKER_PATTERN.fullmatch(text.strip()))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """返回二项比例的 Wilson 区间；没有样本时返回 None。"""

    if total <= 0:
        return None
    ratio = successes / total
    denominator = 1 + z * z / total
    center = (ratio + z * z / (2 * total)) / denominator
    margin = (
        z
        * sqrt((ratio * (1 - ratio) + z * z / (4 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def binary_metrics(
    records: Iterable[tuple[bool, int]],
    *,
    borderline: str = "exclude",
) -> dict[str, object]:
    """按“标签 2 是明确问题、标签 0 是明确正常”计算混淆矩阵。

    边界标签 1 不进入主指标，调用方应另做严格/宽松敏感性分析。
    """

    if borderline not in {"exclude", "negative", "positive"}:
        raise ValueError("borderline 必须为 exclude/negative/positive")
    counts = Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0, "borderline": 0})
    for predicted, label in records:
        if label == 1:
            counts["borderline"] += 1
            if borderline == "exclude":
                continue
            label = 0 if borderline == "negative" else HARD_ISSUE_LABEL
        if label not in {0, HARD_ISSUE_LABEL}:
            raise ValueError(f"人工标签必须为 0/1/2，收到 {label!r}")
        if predicted and label == HARD_ISSUE_LABEL:
            counts["tp"] += 1
        elif predicted:
            counts["fp"] += 1
        elif label == HARD_ISSUE_LABEL:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision_total = tp + fp
    recall_total = tp + fn
    negative_total = fp + tn
    precision = _ratio(tp, precision_total)
    recall = _ratio(tp, recall_total)
    false_positive_rate = _ratio(fp, negative_total)
    false_negative_rate = _ratio(fn, recall_total)
    return {
        **dict(counts),
        "positive_examples": recall_total,
        "negative_examples": negative_total,
        "precision": precision,
        "precision_wilson_95": wilson_interval(tp, precision_total),
        "recall": recall,
        "recall_wilson_95": wilson_interval(tp, recall_total),
        "false_positive_rate": false_positive_rate,
        "false_positive_rate_wilson_95": wilson_interval(fp, negative_total),
        "false_negative_rate": false_negative_rate,
        "false_negative_rate_wilson_95": wilson_interval(fn, recall_total),
    }


def weighted_kappa(first: Iterable[int], second: Iterable[int]) -> float | None:
    """计算 0/1/2 有序标签的线性加权 Cohen's kappa。"""

    first_values = list(first)
    second_values = list(second)
    if len(first_values) != len(second_values):
        raise ValueError("两轮复标长度不一致")
    if not first_values:
        return None
    if any(value not in LABELS for value in [*first_values, *second_values]):
        raise ValueError("复标标签必须为 0/1/2")

    total = len(first_values)
    observed_disagreement = sum(
        abs(left - right) / 2 for left, right in zip(first_values, second_values)
    ) / total
    first_counts = Counter(first_values)
    second_counts = Counter(second_values)
    expected_disagreement = sum(
        (abs(left - right) / 2)
        * (first_counts[left] / total)
        * (second_counts[right] / total)
        for left in LABELS
        for right in LABELS
    )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - observed_disagreement / expected_disagreement


def intrarater_metrics(
    first: Mapping[str, int],
    second: Mapping[str, int],
) -> dict[str, object]:
    """按共同 line_id 评估同一标注者两轮判断的一致性。"""

    common_ids = sorted(set(first) & set(second))
    first_values = [first[line_id] for line_id in common_ids]
    second_values = [second[line_id] for line_id in common_ids]
    exact = sum(a == b for a, b in zip(first_values, second_values))
    binary = sum(
        (a == HARD_ISSUE_LABEL) == (b == HARD_ISSUE_LABEL)
        for a, b in zip(first_values, second_values)
    )
    severe_flips = sum(
        {a, b} == {0, HARD_ISSUE_LABEL}
        for a, b in zip(first_values, second_values)
    )
    total = len(common_ids)
    return {
        "matched_items": total,
        "exact_agreement": _ratio(exact, total),
        "binary_hard_issue_agreement": _ratio(binary, total),
        "severe_flip_rate": _ratio(severe_flips, total),
        "linear_weighted_kappa": weighted_kappa(first_values, second_values),
    }


def score_gate(
    metrics: Mapping[str, object],
    *,
    minimum_positive_examples: int = 20,
    minimum_negative_examples: int = 40,
) -> dict[str, object]:
    """检查单类检测指标；不代表整体评分准入或回炉准入。"""

    positive_examples = int(metrics.get("positive_examples", 0))
    negative_examples = int(metrics.get("negative_examples", 0))
    precision = metrics.get("precision")
    false_positive_rate = metrics.get("false_positive_rate")
    false_negative_rate = metrics.get("false_negative_rate")
    checks = {
        "positive_examples_minimum_met": positive_examples >= minimum_positive_examples,
        "negative_examples_minimum_met": negative_examples >= minimum_negative_examples,
        "precision_at_least_0_80": isinstance(precision, float) and precision >= 0.80,
        "false_positive_rate_at_most_0_10": (
            isinstance(false_positive_rate, float) and false_positive_rate <= 0.10
        ),
        "false_negative_rate_at_most_0_30": (
            isinstance(false_negative_rate, float) and false_negative_rate <= 0.30
        ),
    }
    return {
        "eligible": all(checks.values()),
        "minimum_positive_examples": minimum_positive_examples,
        "minimum_negative_examples": minimum_negative_examples,
        "checks": checks,
    }
