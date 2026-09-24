"""Shadow-evaluate fail-closed minimal-diff Producer constraint repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import ProductionPlan, pydantic_validation_feedback
from scripts.evaluate_producer_constraint_fault_injection import (
    FaultCase,
    FaultManifest,
    load_manifest,
    render_blind_review,
    resolve_cases,
)
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.producer_constraint_guard import (
    ConstraintViolation,
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)
from tools.rag.producer_constraint_patch import (
    FragmentPatch,
    LocatedViolation,
    apply_fragment_patch,
    changed_fields,
    deterministic_patch,
    unresolved_location,
    validate_patched_plan,
)


CASES_PATH = Path("data/miaoxiang/producer_constraint_local_fault_injection.v2.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FragmentReplacement(_StrictModel):
    replacement_fragment: str = Field(min_length=1, max_length=300)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimal_blind_mapping(case_id: str) -> dict[str, str]:
    digest = hashlib.sha256(f"minimal:{case_id}".encode("utf-8")).digest()[0]
    if digest % 2:
        return {"X": "minimal", "Y": "base"}
    return {"X": "base", "Y": "minimal"}


def build_blind_card(
    source_case: Any,
    fault: FaultCase,
    minimal_plan: ProductionPlan,
) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = minimal_blind_mapping(fault.case_id)
    plans = {"base": fault.base_plan, "minimal": minimal_plan}
    return (
        {
            "case_id": fault.review_case_id or fault.case_id,
            "song_request": source_case.user_request_text,
            "song_requirement": source_case.requirement.model_dump(mode="json"),
            "lyrics": source_case.lyric_draft.model_dump(mode="json"),
            "X": plans[mapping["X"]].model_dump(mode="json"),
            "Y": plans[mapping["Y"]].model_dump(mode="json"),
        },
        mapping,
    )


def rebuild_blind_materials(
    manifest: FaultManifest,
    *,
    experiment_dir: Path,
) -> dict[str, Any]:
    summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    completed = {row["case_id"]: row for row in summary.get("completed", [])}
    cards: list[dict[str, Any]] = []
    keys: list[dict[str, str]] = []
    for source_case, fault in resolve_cases(manifest):
        row = completed.get(fault.case_id)
        if row is None:
            raise ValueError(f"minimal_patch_completed_case_missing:{fault.case_id}")
        if row["final_equals_base"]:
            continue
        minimal_plan = ProductionPlan.model_validate(row["final_plan"])
        card, mapping = build_blind_card(source_case, fault, minimal_plan)
        cards.append(card)
        keys.append({"case_id": card["case_id"], **mapping})
    cards_path = experiment_dir / "blind_review_cards.json"
    key_path = experiment_dir / "blind_review_key.json"
    review_path = experiment_dir / "blind_review.md"
    _write_json(cards_path, cards)
    _write_json(key_path, keys)
    review_path.write_text(render_blind_review(cards), encoding="utf-8")
    return {
        "review_case_count": len(cards),
        "cards": str(cards_path),
        "key": str(key_path),
        "review": str(review_path),
    }


def reanalyze_existing_run(
    manifest: FaultManifest,
    *,
    experiment_dir: Path,
) -> dict[str, Any]:
    summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    completed = {row["case_id"]: row for row in summary.get("completed", [])}
    rows: list[dict[str, Any]] = []
    for source_case, fault in resolve_cases(manifest):
        row = completed.get(fault.case_id)
        if row is None:
            raise ValueError(f"minimal_patch_completed_case_missing:{fault.case_id}")
        original = ProductionPlan.model_validate(row["input_plan"])
        final = ProductionPlan.model_validate(row["final_plan"])
        patches = [FragmentPatch.model_validate(item) for item in row["patches"]]
        validation = validate_patched_plan(
            original,
            final,
            patches=patches,
            user_text=source_case.user_request_text,
            constraints=extract_explicit_constraints(source_case.user_request_text),
        )
        rows.append(
            {
                "case_id": fault.case_id,
                "accepted": validation.accepted,
                "validation": validation.model_dump(mode="json"),
                "final_equals_base": final == fault.base_plan,
            }
        )
    result = {
        "schema_version": "1.0",
        "source_summary": str(experiment_dir / "summary.json"),
        "case_count": len(rows),
        "accepted_count": sum(row["accepted"] for row in rows),
        "all_accepted": all(row["accepted"] for row in rows),
        "rows": rows,
        "runtime_agent_connected": False,
    }
    _write_json(experiment_dir / "summary_revalidated.json", result)
    return result


def _hard_violations(plan: ProductionPlan, *, user_text: str):
    constraints = extract_explicit_constraints(user_text)
    violations = check_plan_constraints(
        plan, user_text=user_text, constraints=constraints
    )
    hard, advisory = split_constraint_violations(violations)
    return constraints, hard, advisory


def plan_deterministic_stage(
    plan: ProductionPlan,
    *,
    user_text: str,
) -> tuple[ProductionPlan, list[FragmentPatch], list[ConstraintViolation]]:
    current = plan
    patches: list[FragmentPatch] = []
    constraints = extract_explicit_constraints(user_text)
    for _ in range(8):
        violations = check_plan_constraints(
            current, user_text=user_text, constraints=constraints
        )
        hard, _ = split_constraint_violations(violations)
        if not hard:
            return current, patches, []
        progress = False
        for violation in hard:
            patch = deterministic_patch(
                current,
                violation=violation,
                constraints=constraints,
            )
            if patch is None:
                continue
            current = apply_fragment_patch(current, patch)
            patches.append(patch)
            progress = True
            break
        if not progress:
            return current, patches, hard
    raise ValueError("deterministic_patch_iteration_limit")


def preview_case(
    source_case: Any,
    fault: FaultCase,
    *,
    allow_llm_fragment_shadow: bool = False,
) -> dict[str, Any]:
    intermediate, patches, unresolved = plan_deterministic_stage(
        fault.injected_plan,
        user_text=source_case.user_request_text,
    )
    location: LocatedViolation | None = None
    if len(unresolved) == 1:
        location = unresolved_location(intermediate, unresolved[0])
    if not unresolved:
        action = "deterministic_only"
    elif allow_llm_fragment_shadow and len(unresolved) == 1 and location is not None:
        action = "llm_fragment_once"
    else:
        action = "fail_closed"
    return {
        "case_id": fault.case_id,
        "review_case_id": fault.review_case_id or fault.case_id,
        "fault_family": fault.fault_family,
        "action": action,
        "deterministic_patches": [row.model_dump(mode="json") for row in patches],
        "remaining_hard": [row.model_dump(mode="json") for row in unresolved],
        "llm_location": location.model_dump(mode="json") if location else None,
        "intermediate_plan": intermediate.model_dump(mode="json"),
    }


def build_fragment_messages(
    *,
    user_text: str,
    field_value: str,
    location: LocatedViolation,
    violation: ConstraintViolation,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你只修复一个制作方案片段。返回JSON对象且只能包含replacement_fragment。"
                "替换片段要保持原片段的语法角色和制作功能，只移除明确冲突；"
                "不得新增曲风、乐器、人声身份、BPM、场景或叙事。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户原始请求：{user_text}\n"
                f"目标字段：{location.field}\n"
                f"字段全文：{field_value}\n"
                f"只允许替换的原片段：{location.original_fragment}\n"
                "明确硬冲突："
                + json.dumps(violation.model_dump(mode="json"), ensure_ascii=False)
                + "\n请输出最小替代片段。不要输出完整字段或完整制作方案。"
            ),
        },
    ]


def _call_fragment_repair(
    llm: DeepSeekLLM,
    *,
    user_text: str,
    plan: ProductionPlan,
    location: LocatedViolation,
    violation: ConstraintViolation,
) -> FragmentPatch:
    value = plan.model_dump(mode="json")[location.field]
    if not isinstance(value, str):
        raise ValueError("llm_fragment_target_must_be_text")
    payload = llm.chat_json(
        build_fragment_messages(
            user_text=user_text,
            field_value=value,
            location=location,
            violation=violation,
        ),
        temperature=0.2,
        max_tokens=300,
        operation="producer_constraint_minimal_fragment_repair",
        payload_validator=lambda data: pydantic_validation_feedback(
            FragmentReplacement, data
        ),
    )
    replacement = FragmentReplacement.model_validate(payload).replacement_fragment
    if replacement == location.original_fragment:
        raise ValueError("llm_fragment_replacement_unchanged")
    return FragmentPatch(
        field=location.field,
        original_fragment=location.original_fragment,
        replacement_fragment=replacement,
        violation_code=violation.code,
        concept=violation.concept,
        strategy="llm_fragment",
    )


def run_experiment(
    manifest: FaultManifest,
    *,
    manifest_path: Path,
    output_root: Path,
    allow_llm_fragment_shadow: bool = False,
) -> Path:
    pairs = resolve_cases(manifest)
    experiment_dir = output_root / f"producer-constraint-minimal-patch-v1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    previews = [
        preview_case(
            source,
            fault,
            allow_llm_fragment_shadow=allow_llm_fragment_shadow,
        )
        for source, fault in pairs
    ]
    settings = get_settings()
    _write_json(experiment_dir / "previews.json", previews)
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _file_hash(manifest_path),
            "case_count": len(pairs),
            "repair_policy": "minimal_fragment_fail_closed",
            "llm_fragment_shadow_enabled": allow_llm_fragment_shadow,
            "max_llm_fragment_calls_per_case": 1,
            "audio_generated": False,
            "runtime_agent_connected": False,
            "llm": {
                "provider": "DeepSeek",
                "model": settings.deepseek_model,
                "json_thinking_mode": (
                    "disabled" if settings.llm_json_disable_thinking else "default"
                ),
            },
        },
    )

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    llm = DeepSeekLLM(settings)
    try:
        for source_case, fault in pairs:
            print(f"运行最小差异修复：{fault.case_id}", flush=True)
            case_dir = experiment_dir / "cases" / fault.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                current, patches, unresolved = plan_deterministic_stage(
                    fault.injected_plan,
                    user_text=source_case.user_request_text,
                )
                before_calls = len(llm.telemetry_snapshot())
                if unresolved:
                    if not allow_llm_fragment_shadow:
                        raise ValueError("non_deterministic_fragment_repair_disabled")
                    if len(unresolved) != 1:
                        raise ValueError("multiple_unresolved_hard_violations")
                    location = unresolved_location(current, unresolved[0])
                    if location is None:
                        raise ValueError("unresolved_hard_location_not_unique")
                    patch = _call_fragment_repair(
                        llm,
                        user_text=source_case.user_request_text,
                        plan=current,
                        location=location,
                        violation=unresolved[0],
                    )
                    current = apply_fragment_patch(current, patch)
                    patches.append(patch)
                constraints = extract_explicit_constraints(source_case.user_request_text)
                validation = validate_patched_plan(
                    fault.injected_plan,
                    current,
                    patches=patches,
                    user_text=source_case.user_request_text,
                    constraints=constraints,
                )
                records = llm.telemetry_snapshot()[before_calls:]
                if not validation.accepted:
                    raise ValueError(f"minimal_patch_rejected:{validation.reason}")
                report = {
                    "case_id": fault.case_id,
                    "review_case_id": fault.review_case_id or fault.case_id,
                    "fault_family": fault.fault_family,
                    "status": "accepted",
                    "input_plan": fault.injected_plan.model_dump(mode="json"),
                    "final_plan": current.model_dump(mode="json"),
                    "base_plan": fault.base_plan.model_dump(mode="json"),
                    "patches": [row.model_dump(mode="json") for row in patches],
                    "validation": validation.model_dump(mode="json"),
                    "final_equals_base": current == fault.base_plan,
                    "input_to_final_changed_fields": changed_fields(
                        fault.injected_plan, current
                    ),
                    "base_to_final_changed_fields": changed_fields(
                        fault.base_plan, current
                    ),
                    "llm_telemetry": DeepSeekLLM.summarize_call_records(records),
                }
                _write_json(case_dir / "result.json", report)
                completed.append(report)
            except Exception as error:
                fail_closed = str(error) == "non_deterministic_fragment_repair_disabled"
                failure = {
                    "case_id": fault.case_id,
                    "status": "fail_closed" if fail_closed else "repair_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "fallback_used": False,
                    "failure_mode": "non_deterministic_repair" if fail_closed else "repair_error",
                }
                _write_json(case_dir / "failure.json", failure)
                failures.append(failure)
    finally:
        telemetry = DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot())
        llm.close()

    _write_json(
        experiment_dir / "summary.json",
        {
            "completed_case_count": len(completed),
            "failed_case_count": len(failures),
            "fail_closed_case_count": sum(
                row["status"] == "fail_closed" for row in failures
            ),
            "final_equals_base_count": sum(row["final_equals_base"] for row in completed),
            "deterministic_only_count": sum(
                not row["llm_telemetry"]["successful_calls"] for row in completed
            ),
            "llm_fragment_case_count": sum(
                bool(row["llm_telemetry"]["successful_calls"]) for row in completed
            ),
            "completed": completed,
            "failures": failures,
            "llm_telemetry": telemetry,
            "fallback_policy": "deterministic_only_else_fail_closed",
            "llm_fragment_shadow_enabled": allow_llm_fragment_shadow,
            "runtime_agent_connected": False,
            "runtime_injection_changed": False,
            "audio_generated": False,
        },
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行Producer最小差异片段修复影子实验。"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--rebuild-blind-from",
        type=Path,
        help="从既有最小修复结果离线生成非等价案例盲包，不调用模型",
    )
    parser.add_argument(
        "--reanalyze-from",
        type=Path,
        help="用当前验收器离线复核既有运行结果，不调用模型",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="运行确定性补丁；无法确定性修复时失败关闭，不生成音频",
    )
    parser.add_argument(
        "--allow-llm-fragment-shadow",
        action="store_true",
        help="仅研究历史片段模型分支；默认关闭，不代表准入",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.cases)
    if args.reanalyze_from is not None:
        print(
            json.dumps(
                reanalyze_existing_run(
                    manifest,
                    experiment_dir=args.reanalyze_from,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if args.rebuild_blind_from is not None:
        print(
            json.dumps(
                rebuild_blind_materials(
                    manifest,
                    experiment_dir=args.rebuild_blind_from,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    previews = [
        preview_case(
            source_case,
            fault,
            allow_llm_fragment_shadow=args.allow_llm_fragment_shadow,
        )
        for source_case, fault in resolve_cases(manifest)
    ]
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(previews),
                    "deterministic_only_count": sum(
                        row["action"] == "deterministic_only" for row in previews
                    ),
                    "planned_llm_fragment_calls": sum(
                        row["action"] == "llm_fragment_once" for row in previews
                    ),
                    "fail_closed_count": sum(
                        row["action"] == "fail_closed" for row in previews
                    ),
                    "audio_generated": False,
                    "runtime_agent_connected": False,
                    "previews": previews,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    output = run_experiment(
        manifest,
        manifest_path=args.cases,
        output_root=args.out,
        allow_llm_fragment_shadow=args.allow_llm_fragment_shadow,
    )
    print(f"实验输出：{output}", flush=True)


if __name__ == "__main__":
    main()
