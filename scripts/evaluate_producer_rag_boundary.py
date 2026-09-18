"""Run repeated high-risk boundary checks for the conservative style-RAG gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.schemas import LyricDraft, ProductionPlan, SongRequirement
from scripts.evaluate_producer_rag_gated_ab import (
    blind_mapping,
    build_blind_card,
    build_gate_context,
    build_variant_messages,
    call_producer,
    render_blind_review,
    variant_order,
)
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.style_retrieval_policy import Action, decide_style_retrieval


CASES_PATH = Path("data/miaoxiang/producer_rag_boundary_validation.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
VARIANTS = ("baseline", "gated")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class BoundaryGenerationFamily(_StrictModel):
    family_id: str = Field(min_length=1)
    case_type: Literal["explicit_multi_neighbor", "optional_rap_pair"]
    repeat_count: int = Field(ge=2, le=2)
    user_request_text: str = Field(min_length=5)
    expected_action: Literal["multi_candidate"]
    expected_candidate_ids: list[str] = Field(min_length=2, max_length=2)
    requirement: SongRequirement
    lyric_draft: LyricDraft

    @property
    def case_id(self) -> str:
        """Expose the expanded sample identifier to shared blind helpers."""

        return self.family_id


class BoundaryOfflineCase(_StrictModel):
    case_id: str = Field(min_length=1)
    case_type: Literal["hard_constraint_conflict", "broad_low_confidence"]
    user_request_text: str = Field(min_length=5)
    expected_action: Literal["no_rag"]
    expected_reason: str = Field(min_length=1)
    expected_candidate_ids: list[str] = Field(max_length=0)


class BoundaryManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_rag_boundary_validation"]
    frozen_at: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)
    generation_families: list[BoundaryGenerationFamily] = Field(
        min_length=2, max_length=2
    )
    offline_cases: list[BoundaryOfflineCase] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_fixed_mix(self) -> "BoundaryManifest":
        family_ids = [row.family_id for row in self.generation_families]
        offline_ids = [row.case_id for row in self.offline_cases]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("boundary_family_ids_must_be_unique")
        if len(offline_ids) != len(set(offline_ids)):
            raise ValueError("boundary_offline_ids_must_be_unique")
        if {row.case_type for row in self.generation_families} != {
            "explicit_multi_neighbor",
            "optional_rap_pair",
        }:
            raise ValueError("boundary_generation_mix_mismatch")
        if {row.case_type for row in self.offline_cases} != {
            "hard_constraint_conflict",
            "broad_low_confidence",
        }:
            raise ValueError("boundary_offline_mix_mismatch")
        return self


def load_manifest(path: Path = CASES_PATH) -> BoundaryManifest:
    return BoundaryManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expanded_generation_cases(
    manifest: BoundaryManifest,
) -> list[BoundaryGenerationFamily]:
    expanded: list[BoundaryGenerationFamily] = []
    for family in manifest.generation_families:
        for replicate in range(1, family.repeat_count + 1):
            expanded.append(
                family.model_copy(
                    update={"family_id": f"{family.family_id}-r{replicate}"}
                )
            )
    return expanded


def validate_gate_contracts(manifest: BoundaryManifest) -> dict[str, Any]:
    generation: list[dict[str, Any]] = []
    for family in manifest.generation_families:
        decision = decide_style_retrieval(family.user_request_text)
        if decision.action != family.expected_action:
            raise ValueError(
                f"generation_action_mismatch:{family.family_id}:{decision.action}"
            )
        if decision.candidate_ids != family.expected_candidate_ids:
            raise ValueError(
                f"generation_candidates_mismatch:{family.family_id}:"
                f"{decision.candidate_ids}"
            )
        context = build_gate_context(decision)
        baseline = build_variant_messages(
            family, variant="baseline", decision=decision, context=context
        )
        gated = build_variant_messages(
            family, variant="gated", decision=decision, context=context
        )
        if baseline == gated:
            raise ValueError(f"generation_prompts_unexpectedly_equal:{family.family_id}")
        generation.append(
            {
                "family_id": family.family_id,
                "action": decision.action,
                "candidate_ids": decision.candidate_ids,
                "prompts_identical": False,
            }
        )

    offline: list[dict[str, Any]] = []
    for case in manifest.offline_cases:
        decision = decide_style_retrieval(case.user_request_text)
        if decision.action != case.expected_action:
            raise ValueError(
                f"offline_action_mismatch:{case.case_id}:{decision.action}"
            )
        if decision.reason != case.expected_reason:
            raise ValueError(
                f"offline_reason_mismatch:{case.case_id}:{decision.reason}"
            )
        if decision.candidate_ids != case.expected_candidate_ids:
            raise ValueError(
                f"offline_candidates_mismatch:{case.case_id}:"
                f"{decision.candidate_ids}"
            )
        offline.append(
            {
                "case_id": case.case_id,
                "action": decision.action,
                "reason": decision.reason,
                "candidate_ids": decision.candidate_ids,
                "deepseek_calls": 0,
            }
        )
    return {"generation_families": generation, "offline_cases": offline}


def _render_boundary_review(cards: list[dict[str, Any]]) -> str:
    review = render_blind_review(cards)
    review = review.replace(
        "# Producer 制作方案第二轮匿名复核",
        "# Producer 高风险边界重复采样匿名复核",
        1,
    )
    marker = "只评价歌曲制作方案文本，不评价音频。X/Y 身份已隐藏。\n"
    note = (
        "同一主题会出现两次独立采样；请逐组单独评价，不要根据另一组猜测身份。\n"
    )
    return review.replace(marker, marker + note, 1)


def run_experiment(
    manifest: BoundaryManifest,
    *,
    output_root: Path,
) -> Path:
    gate_contracts = validate_gate_contracts(manifest)
    expanded = expanded_generation_cases(manifest)
    experiment_dir = output_root / f"producer-rag-boundary-v1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(experiment_dir / "gate_contracts.json", gate_contracts)
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(CASES_PATH),
            "case_manifest_sha256": _manifest_hash(CASES_PATH),
            "generation_family_count": len(manifest.generation_families),
            "replicate_pair_count": len(expanded),
            "offline_no_rag_count": len(manifest.offline_cases),
            "variants": list(VARIANTS),
            "planned_deepseek_calls": len(expanded) * 2,
            "audio_generated": False,
            "runtime_agent_connected": False,
            "runtime_injection_changed": False,
            "llm": {
                "provider": "DeepSeek",
                "model": settings.deepseek_model,
                "max_retries": settings.llm_max_retries,
                "json_thinking_mode": (
                    "disabled" if settings.llm_json_disable_thinking else "default"
                ),
            },
            "limitations": [
                "Two families with two replicates are development evidence, not statistics.",
                "The same reviewer and text-only modality remain limitations.",
                "No audio or backend style following is evaluated.",
            ],
        },
    )

    blind_cards: list[dict[str, Any]] = []
    blind_keys: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    llm = DeepSeekLLM(settings)
    try:
        for sample_case in expanded:
            sample_id = sample_case.family_id
            base_family_id = sample_id.rsplit("-r", 1)[0]
            print(f"运行边界样本：{sample_id}", flush=True)
            case_dir = experiment_dir / "cases" / sample_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                decision = decide_style_retrieval(sample_case.user_request_text)
                context = build_gate_context(decision)
                _write_json(
                    case_dir / "input.json", sample_case.model_dump(mode="json")
                )
                _write_json(
                    case_dir / "gate_decision.json", decision.model_dump(mode="json")
                )
                _write_json(case_dir / "experimental_context.json", context)
                plans: dict[str, ProductionPlan] = {}
                call_telemetry: dict[str, Any] = {}
                for variant in variant_order(sample_id):
                    before = len(llm.telemetry_snapshot())
                    plans[variant] = call_producer(
                        llm,
                        sample_case,
                        variant=variant,
                        decision=decision,
                        context=context,
                    )
                    records = llm.telemetry_snapshot()[before:]
                    call_telemetry[variant] = DeepSeekLLM.summarize_call_records(records)
                    _write_json(
                        case_dir / f"{variant}.json",
                        plans[variant].model_dump(mode="json"),
                    )
                card, mapping = build_blind_card(sample_case, plans)
                card["family_id"] = base_family_id
                card["replicate"] = int(sample_id.rsplit("-r", 1)[1])
                blind_cards.append(card)
                blind_keys.append(
                    {
                        "case_id": sample_id,
                        "family_id": base_family_id,
                        **mapping,
                    }
                )
                completed.append(
                    {
                        "case_id": sample_id,
                        "family_id": base_family_id,
                        "gate_action": decision.action,
                        "candidate_ids": decision.candidate_ids,
                        "call_order": list(variant_order(sample_id)),
                        "contracts_valid": True,
                        "telemetry": call_telemetry,
                    }
                )
            except Exception as error:
                failure = {
                    "case_id": sample_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _write_json(case_dir / "failure.json", failure)
                print(f"边界样本失败：{sample_id}", flush=True)
    finally:
        telemetry = DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot())
        llm.close()

    _write_json(experiment_dir / "blind_review_cards.json", blind_cards)
    _write_json(experiment_dir / "blind_review_key.json", blind_keys)
    (experiment_dir / "blind_review.md").write_text(
        _render_boundary_review(blind_cards), encoding="utf-8"
    )
    _write_json(
        experiment_dir / "summary.json",
        {
            "completed_pair_count": len(completed),
            "failed_pair_count": len(failures),
            "expected_logical_calls": len(expanded) * 2,
            "completed": completed,
            "failures": failures,
            "offline_no_rag_contracts": gate_contracts["offline_cases"],
            "llm_telemetry": telemetry,
            "human_review_status": "pending",
            "method_decision": None,
        },
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行Producer风格RAG高风险边界重复采样验证。"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute", action="store_true", help="确认调用8次真实DeepSeek；不生成音频"
    )
    args = parser.parse_args()
    manifest = load_manifest(args.cases)
    contracts = validate_gate_contracts(manifest)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "generation_family_count": len(manifest.generation_families),
                    "replicate_pair_count": len(expanded_generation_cases(manifest)),
                    "planned_deepseek_calls": len(expanded_generation_cases(manifest))
                    * 2,
                    "offline_deepseek_calls": 0,
                    "audio_generated": False,
                    "runtime_agent_connected": False,
                    "gate_contracts": contracts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    output = run_experiment(manifest, output_root=args.out)
    print(f"实验输出：{output}", flush=True)


if __name__ == "__main__":
    main()
