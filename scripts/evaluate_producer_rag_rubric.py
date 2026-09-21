"""Evaluate the clarified Producer style-RAG review rubric on six new cases.

This remains an isolated text experiment. It reuses the conservative shadow
gate and never changes the runtime Producer, LangGraph, Web route, or audio
backend.
"""

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
    build_blind_card,
    build_gate_context,
    build_variant_messages,
    call_producer,
    render_blind_review,
    variant_order,
)
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.style_retrieval_policy import decide_style_retrieval


CASES_PATH = Path("data/miaoxiang/producer_rag_rubric_validation.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
VARIANTS = ("baseline", "gated")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RubricValidationCase(_StrictModel):
    case_id: str = Field(min_length=1)
    risk_focus: Literal[
        "style_native_detail",
        "unsupported_vocal_expansion",
        "hard_instrumental_preservation",
        "soft_default_override",
        "soft_gender_and_timbre_override",
        "explicit_instrument_override",
    ]
    user_request_text: str = Field(min_length=5)
    expected_action: Literal["single_style"]
    expected_candidate_ids: list[str] = Field(min_length=1, max_length=1)
    requirement: SongRequirement
    lyric_draft: LyricDraft


class RubricValidationManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_rag_rubric_validation"]
    frozen_at: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)
    cases: list[RubricValidationCase] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def validate_fixed_set(self) -> "RubricValidationManifest":
        case_ids = [case.case_id for case in self.cases]
        risk_focuses = [case.risk_focus for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("rubric_validation_case_ids_must_be_unique")
        if len(risk_focuses) != len(set(risk_focuses)):
            raise ValueError("rubric_validation_risk_focuses_must_be_unique")
        return self


def load_manifest(path: Path = CASES_PATH) -> RubricValidationManifest:
    return RubricValidationManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_gate_contracts(
    manifest: RubricValidationManifest,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for case in manifest.cases:
        decision = decide_style_retrieval(case.user_request_text)
        if decision.action != case.expected_action:
            raise ValueError(
                f"gate_action_mismatch:{case.case_id}:{decision.action}"
            )
        if decision.candidate_ids != case.expected_candidate_ids:
            raise ValueError(
                f"gate_candidates_mismatch:{case.case_id}:{decision.candidate_ids}"
            )
        context = build_gate_context(decision)
        baseline = build_variant_messages(
            case, variant="baseline", decision=decision, context=context
        )
        gated = build_variant_messages(
            case, variant="gated", decision=decision, context=context
        )
        if baseline == gated:
            raise ValueError(f"gated_prompt_unexpectedly_equal:{case.case_id}")
        contracts.append(
            {
                "case_id": case.case_id,
                "risk_focus": case.risk_focus,
                "action": decision.action,
                "candidate_ids": decision.candidate_ids,
                "prompts_identical": False,
            }
        )
    return contracts


def render_rubric_review(cards: list[dict[str, Any]]) -> str:
    review = render_blind_review(cards).replace(
        "# Producer 制作方案第二轮匿名复核",
        "# Producer 制作方案评审口径 v2 匿名复核",
        1,
    )
    old = (
        "每组分别评价：风格/需求契合度、制作可执行性、内部一致性（均为1-5），\n"
        "以及是否过度约束、是否有无依据扩写，最后选择 X / Y / 平局。"
    )
    new = "\n".join(
        [
            "每组分别评价：风格/需求契合度、制作可执行性、内部一致性（均为1-5），",
            "以及是否过度约束、是否有无依据扩写，最后选择 X / Y / 平局。",
            "风格原生且不改变用户要求的制作细节不自动算扩写；只有非必要细节被写成不可替换的强制实现才算过度约束。",
            "若偏好三项总分更低的一方，必须指出足以覆盖分差的决定性问题。",
            "中文‘混声’指 mix voice / chest-head mix，不是 mixed-gender vocal。",
        ]
    )
    return review.replace(old, new, 1)


def run_experiment(
    manifest: RubricValidationManifest,
    *,
    manifest_path: Path,
    output_root: Path,
) -> Path:
    contracts = validate_gate_contracts(manifest)
    experiment_dir = output_root / f"producer-rag-rubric-v2-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(experiment_dir / "gate_contracts.json", contracts)
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _manifest_hash(manifest_path),
            "rubric": manifest.rubric,
            "case_count": len(manifest.cases),
            "variants": list(VARIANTS),
            "planned_deepseek_calls": len(manifest.cases) * 2,
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
                "Six single-sample pairs are rubric calibration, not statistics.",
                "Text plans do not prove audio quality or backend control.",
                "Web reviewers and exact model versions are not controlled by this script.",
            ],
        },
    )

    blind_cards: list[dict[str, Any]] = []
    blind_keys: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    llm = DeepSeekLLM(settings)
    try:
        for case in manifest.cases:
            print(f"运行口径校准案例：{case.case_id}", flush=True)
            case_dir = experiment_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                decision = decide_style_retrieval(case.user_request_text)
                context = build_gate_context(decision)
                _write_json(case_dir / "input.json", case.model_dump(mode="json"))
                _write_json(
                    case_dir / "gate_decision.json", decision.model_dump(mode="json")
                )
                _write_json(case_dir / "experimental_context.json", context)
                plans: dict[str, ProductionPlan] = {}
                call_telemetry: dict[str, Any] = {}
                for variant in variant_order(case.case_id):
                    before = len(llm.telemetry_snapshot())
                    plans[variant] = call_producer(
                        llm,
                        case,
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
                card, mapping = build_blind_card(case, plans)
                blind_cards.append(card)
                blind_keys.append(
                    {
                        "case_id": case.case_id,
                        "risk_focus": case.risk_focus,
                        **mapping,
                    }
                )
                completed.append(
                    {
                        "case_id": case.case_id,
                        "risk_focus": case.risk_focus,
                        "gate_action": decision.action,
                        "candidate_ids": decision.candidate_ids,
                        "call_order": list(variant_order(case.case_id)),
                        "contracts_valid": True,
                        "telemetry": call_telemetry,
                    }
                )
            except Exception as error:
                failure = {
                    "case_id": case.case_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _write_json(case_dir / "failure.json", failure)
                print(f"口径校准案例失败：{case.case_id}", flush=True)
    finally:
        telemetry = DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot())
        llm.close()

    _write_json(experiment_dir / "blind_review_cards.json", blind_cards)
    _write_json(experiment_dir / "blind_review_key.json", blind_keys)
    (experiment_dir / "blind_review.md").write_text(
        render_rubric_review(blind_cards), encoding="utf-8"
    )
    _write_json(
        experiment_dir / "summary.json",
        {
            "completed_case_count": len(completed),
            "failed_case_count": len(failures),
            "expected_logical_calls": len(manifest.cases) * 2,
            "completed": completed,
            "failures": failures,
            "llm_telemetry": telemetry,
            "multi_model_review_status": "pending",
            "method_decision": None,
        },
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行Producer风格RAG评审口径v2的六案例匿名文本对照。"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute", action="store_true", help="确认调用12次真实DeepSeek；不生成音频"
    )
    args = parser.parse_args()
    manifest = load_manifest(args.cases)
    contracts = validate_gate_contracts(manifest)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(manifest.cases),
                    "planned_deepseek_calls": len(manifest.cases) * 2,
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
    output = run_experiment(
        manifest,
        manifest_path=args.cases,
        output_root=args.out,
    )
    print(f"实验输出：{output}", flush=True)


if __name__ == "__main__":
    main()
