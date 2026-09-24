"""Build an independent field-position holdout for deterministic patching."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_producer_constraint_fault_injection import FaultManifest


SOURCE_MANIFEST = Path("data/miaoxiang/producer_constraint_severity_holdout.v1.json")
SOURCE_EXPERIMENT = Path(
    "data/output/evaluations/"
    "producer-constraint-severity-v1-a65fa71a74c4405ca157fe4f31c2f669/summary.json"
)
DEFAULT_OUTPUT = Path(
    "data/miaoxiang/producer_constraint_minimal_patch_field_position_holdout.v1.json"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_hash(case_id: str) -> str:
    return hashlib.sha256(f"minimal-field-position-holdout:{case_id}".encode()).hexdigest()


def build_manifest(
    *,
    source_manifest_path: Path = SOURCE_MANIFEST,
    source_experiment_path: Path = SOURCE_EXPERIMENT,
) -> dict[str, Any]:
    source = _read_json(source_manifest_path)
    source_cases = {row["case_id"]: row for row in source["cases"]}
    summary = _read_json(source_experiment_path)
    plans = {row["case_id"]: row["initial_plan"] for row in summary["completed"]}

    definitions = [
        {
            "case_id": "ho01-vina-style-gender",
            "source_case_id": "h06-vina-female-laundry",
            "fault_family": "wrong_vocal_gender",
            "expected_hard_code": "forbidden_term_present",
            "target_field": "style_prompt",
            "original_fragment": "Chinese clear female vocal",
            "injected_fragment": "Chinese clear male vocal",
        },
        {
            "case_id": "ho02-dnb-style-rap",
            "source_case_id": "h02-dnb-sung-night-bus",
            "fault_family": "forbidden_rap_mc_slogans",
            "expected_hard_code": "forbidden_term_present",
            "target_field": "style_prompt",
            "original_fragment": "no rap, no MC, no sampled vocal shouts",
            "injected_fragment": "rap, MC, sampled vocal shouts",
        },
        {
            "case_id": "ho03-techno-arrangement-vocal",
            "source_case_id": "h03-techno-instrumental-greenhouse",
            "fault_family": "vocal_material_in_instrumental",
            "expected_hard_code": "forbidden_term_present",
            "target_field": "arrangement_note",
            "original_fragment": "只留阀门点击与低频余响",
            "injected_fragment": "加入短促失真人声与口白",
        },
        {
            "case_id": "ho04-deep-house-arrangement-bpm",
            "source_case_id": "h05-deep-house-flex-bpm-ferry",
            "fault_family": "hard_fixed_bpm",
            "expected_hard_code": "exact_bpm_written_as_hard_requirement",
            "target_field": "arrangement_note",
            "original_fragment": "主歌稀疏",
            "injected_fragment": "主歌稀疏，must stay at 123 BPM",
        },
    ]

    cases: list[dict[str, Any]] = []
    for definition in definitions:
        source_case_id = definition["source_case_id"]
        base = plans[source_case_id]
        field = definition["target_field"]
        original = definition["original_fragment"]
        replacement = definition["injected_fragment"]
        value = base[field]
        if not isinstance(value, str) or value.count(original) != 1:
            raise ValueError(
                f"holdout_fragment_not_unique:{definition['case_id']}:{field}"
            )
        injected = dict(base)
        injected[field] = value.replace(original, replacement, 1)
        cases.append(
            {
                **{
                    key: value
                    for key, value in definition.items()
                    if key != "expected_hard_code"
                },
                "review_case_id": definition["case_id"],
                "source_artifact_sha256": _artifact_hash(source_case_id),
                "expected_injected_hard_codes": [definition["expected_hard_code"]],
                "base_plan": base,
                "injected_plan": injected,
            }
        )

    payload = {
        "schema_version": "1.0",
        "kind": "producer_constraint_local_fault_injection",
        "injection_scope": "single_field_single_fragment",
        "frozen_at": "2026-09-24",
        "source_manifest": str(source_manifest_path),
        "source_experiment": str(source_experiment_path),
        "rubric": "docs/producer-rag-review-rubric-v2.1.md",
        "max_constraint_repairs": 1,
        "notes": [
            "独立holdout复用未参与最小片段方法设计的用户请求，但改变故障出现的字段位置。",
            "每份故障稿只在一个字段替换一个片段，其他字段逐字段相同。",
            "本清单只验证确定性补丁的定位、重建和失败关闭契约，不调用模型或音频。",
        ],
        "cases": cases,
    }
    FaultManifest.model_validate(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="生成Producer确定性补丁字段位置holdout。")
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--source-experiment", type=Path, default=SOURCE_EXPERIMENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest(
        source_manifest_path=args.source_manifest,
        source_experiment_path=args.source_experiment,
    )
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.out), "case_count": len(payload["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
