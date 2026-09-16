"""Dry-run validation and production-admission report for style-card RAG data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tools.rag.schemas import (
    SourceAdmissionPolicy,
    SourceRecord,
    StyleCard,
    production_admission,
)


DEFAULT_ADMISSION_AUDIT_PATH = Path("data/miaoxiang/style_templates.v2.admission-audit.json")


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            rows.append((line_number, json.loads(line)))
    return rows


def _error_text(error: ValidationError) -> list[str]:
    issues: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ()))
        code = str(item.get("type", "validation_error"))
        issues.append(f"{location}:{code}" if location else code)
    return issues


def validate_dataset(
    cards_path: Path,
    sources_path: Path,
    admission_audit_path: Path = DEFAULT_ADMISSION_AUDIT_PATH,
) -> dict[str, Any]:
    source_rows = _read_jsonl(sources_path)
    card_rows = _read_jsonl(cards_path)
    source_models: dict[str, SourceRecord] = {}
    source_errors: list[dict[str, Any]] = []
    source_duplicates: list[str] = []
    admission_policy_errors: list[str] = []
    try:
        audit_payload = json.loads(admission_audit_path.read_text(encoding="utf-8"))
        admission_policy = SourceAdmissionPolicy.model_validate(
            audit_payload["source_use_admission"]
        )
    except (OSError, KeyError, json.JSONDecodeError, ValidationError) as error:
        admission_policy = SourceAdmissionPolicy(
            schema_version="fallback",
            default_status="blocked",
            policy_note="准入审计缺失或无效时 fail-closed。",
            cards={},
        )
        admission_policy_errors.append(type(error).__name__)
    for line_number, payload in source_rows:
        try:
            source = SourceRecord.model_validate(payload)
        except ValidationError as error:
            source_errors.append({"line": line_number, "issues": _error_text(error)})
            continue
        if source.source_id in source_models:
            source_duplicates.append(source.source_id)
        source_models[source.source_id] = source

    card_models: dict[str, StyleCard] = {}
    card_duplicates: list[str] = []
    card_results: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for line_number, payload in card_rows:
        card_id = str(payload.get("id", f"<line-{line_number}>"))
        try:
            card = StyleCard.model_validate(payload)
        except ValidationError as error:
            issues = _error_text(error)
            card_results.append(
                {
                    "line": line_number,
                    "card_id": card_id,
                    "valid": False,
                    "errors": issues,
                    "warnings": [],
                    "local_demo_eligible": False,
                    "local_demo_block_reasons": ["schema_validation_error"],
                    "public_product_eligible": False,
                    "public_product_block_reasons": ["schema_validation_error"],
                }
            )
            reason_counts["schema_validation_error"] += 1
            continue
        if card.id in card_models:
            card_duplicates.append(card.id)
        card_models[card.id] = card
        source = source_models.get(card.source_id)
        errors: list[str] = []
        warnings: list[str] = []
        if source is None:
            errors.append("source_id_not_in_catalog")
        else:
            if source.source_id != card.source_id:
                errors.append("source_id_mismatch")
            if card.source_title != source.title:
                errors.append("source_title_mismatch")
            if card.source_url != source.url:
                errors.append("source_url_mismatch")
            missing_supplemental = [
                row.source_id
                for row in card.supplemental_sources
                if row.source_id not in source_models
            ]
            if missing_supplemental:
                errors.append(
                    "supplemental_source_ids_not_in_catalog:" + ",".join(sorted(missing_supplemental))
                )
        if card.tempo_bpm_min is not None and not card.tempo_usage_note:
            warnings.append("numeric_tempo_should_have_tempo_usage_note")
        if card.evidence_depth == "supplemented_open_reference":
            warnings.append("supplemented_reference_requires_explicit_admission_review")
        related_source_ids = [card.source_id] + [
            row.source_id for row in card.supplemental_sources
        ]
        source_admission_statuses = {
            source_id: (
                admission_policy.sources[source_id].status
                if source_id in admission_policy.sources
                else admission_policy.default_status
            )
            for source_id in related_source_ids
        }
        card_admission = admission_policy.cards.get(card.id)
        card_admission_status = (
            card_admission.status
            if card_admission is not None
            else admission_policy.default_status
        )
        local_demo_eligible, local_demo_reasons = production_admission(
            card,
            source,
            source_admission_statuses=list(source_admission_statuses.values()),
            card_admission_status=card_admission_status,
            requested_scope="local_demo",
            contract_errors=errors,
        )
        public_product_eligible, public_product_reasons = production_admission(
            card,
            source,
            source_admission_statuses=list(source_admission_statuses.values()),
            card_admission_status=card_admission_status,
            requested_scope="public_product",
            contract_errors=errors,
        )
        if (
            card.card_role == "composite_query_template"
            and "not_a_direct_style_candidate" in local_demo_reasons
        ):
            warnings.append("composite_excluded_from_direct_candidates")
        for reason in local_demo_reasons:
            reason_counts[f"local_demo:{reason}"] += 1
        for reason in public_product_reasons:
            reason_counts[f"public_product:{reason}"] += 1
        card_results.append(
            {
                "line": line_number,
                "card_id": card.id,
                "valid": not errors,
                "errors": errors,
                "warnings": warnings,
                "review_status": card.review_status,
                "card_role": card.card_role,
                "retrieval_scope": card.retrieval_scope,
                "source_id": card.source_id,
                "source_admission_statuses": source_admission_statuses,
                "card_admission_status": card_admission_status,
                "local_demo_eligible": local_demo_eligible,
                "local_demo_block_reasons": local_demo_reasons,
                "public_product_eligible": public_product_eligible,
                "public_product_block_reasons": public_product_reasons,
            }
        )

    unknown_admission_source_ids = sorted(
        set(admission_policy.sources) - set(source_models)
    )
    unknown_admission_card_ids = sorted(
        set(admission_policy.cards) - set(card_models)
    )
    admission_policy_errors.extend(
        f"unknown_admission_source_id:{source_id}"
        for source_id in unknown_admission_source_ids
    )
    admission_policy_errors.extend(
        f"unknown_admission_card_id:{card_id}"
        for card_id in unknown_admission_card_ids
    )

    if card_duplicates:
        reason_counts["duplicate_card_id"] += len(card_duplicates)
        duplicate_set = set(card_duplicates)
        for row in card_results:
            if row["card_id"] in duplicate_set:
                row["local_demo_eligible"] = False
                row["public_product_eligible"] = False
                row["local_demo_block_reasons"].append("duplicate_card_id")
                row["public_product_block_reasons"].append("duplicate_card_id")
    if source_duplicates:
        duplicate_source_set = set(source_duplicates)
        for row in card_results:
            if row.get("source_id") in duplicate_source_set:
                row["local_demo_eligible"] = False
                row["public_product_eligible"] = False
                row["local_demo_block_reasons"].append("duplicate_source_id")
                row["public_product_block_reasons"].append("duplicate_source_id")
        reason_counts["duplicate_source_id"] += len(source_duplicates)
    valid_card_results = [row for row in card_results if row["valid"]]
    local_demo_eligible_ids = [
        row["card_id"] for row in card_results if row["local_demo_eligible"]
    ]
    public_product_eligible_ids = [
        row["card_id"] for row in card_results if row["public_product_eligible"]
    ]
    error_count = (
        len(source_errors)
        + len(admission_policy_errors)
        + sum(len(row["errors"]) for row in card_results)
    )
    error_count += len(card_duplicates) + len(source_duplicates)
    return {
        "validation_id": "style-rag-dry-run-v1",
        "validation_mode": "dry_run",
        "cards_path": str(cards_path),
        "sources_path": str(sources_path),
        "admission_audit_path": str(admission_audit_path),
        "card_count": len(card_rows),
        "source_count": len(source_rows),
        "valid_card_count": len(valid_card_results),
        "error_count": error_count,
        "warning_count": sum(len(row["warnings"]) for row in card_results),
        "local_demo_eligible_count": len(local_demo_eligible_ids),
        "local_demo_eligible_card_ids": local_demo_eligible_ids,
        "public_product_eligible_count": len(public_product_eligible_ids),
        "public_product_eligible_card_ids": public_product_eligible_ids,
        "blocked_reason_counts": dict(sorted(reason_counts.items())),
        "duplicate_card_ids": sorted(set(card_duplicates)),
        "duplicate_source_ids": sorted(set(source_duplicates)),
        "source_errors": source_errors,
        "admission_policy_errors": admission_policy_errors,
        "source_admission_default_status": admission_policy.default_status,
        "card_results": card_results,
        "review_status_changed": False,
        "production_index_changed": False,
        "interpretation": "dry-run 分别验证本地演示与公开产品准入，不修改卡片状态或创建 collection。",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, default=Path("data/miaoxiang/style_templates.v2.jsonl"))
    parser.add_argument("--sources", type=Path, default=Path("data/miaoxiang/source_catalog.jsonl"))
    parser.add_argument("--admission-audit", type=Path, default=DEFAULT_ADMISSION_AUDIT_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/output/evaluations/style-rag-dry-run-v1"))
    args = parser.parse_args()

    report = validate_dataset(args.cards, args.sources, args.admission_audit)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "validation_id", "validation_mode", "card_count", "source_count",
        "valid_card_count", "error_count", "warning_count",
        "local_demo_eligible_count", "public_product_eligible_count",
        "review_status_changed",
        "production_index_changed",
    )}, ensure_ascii=False, indent=2))
    print(json.dumps(report["blocked_reason_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
