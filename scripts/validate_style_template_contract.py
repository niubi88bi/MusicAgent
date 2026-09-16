"""Validate the draft style-card contract without changing production data.

The project has not implemented ``tools/rag/schemas.py`` yet.  This module is
therefore an explicit shadow contract: it checks the current JSONL records and
reports whether they are ready for a future production schema, but it does not
create an index or change admission state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


BASE_REQUIRED_FIELDS = {
    "id",
    "schema_version",
    "kind",
    "genre",
    "subgenres",
    "moods",
    "tempo_bpm_min",
    "tempo_bpm_max",
    "reference_bpm_examples",
    "instruments",
    "vocal_options",
    "melody",
    "dynamics",
    "structure_notes",
    "official_prompt_dimensions",
    "official_facts",
    "evidence_depth",
    "summary",
    "prompt_template",
    "retrieval_terms",
    "backend_tags",
    "backend_tag_source",
    "supported_backends",
    "backend_validation",
    "source_type",
    "source_id",
    "source_title",
    "source_url",
    "accessed_at",
    "license_or_usage_note",
    "derivation_note",
    "content_sha256",
    "content_hash_scope",
    "review_status",
    "version",
    "card_role",
    "retrieval_scope",
}

VOICE_FIELDS = {
    "vocal_type",
    "rap_default",
    "instrumental_default",
    "default_language",
}
VOICE_TYPES = {
    "female_sung",
    "male_sung",
    "sung_any",
    "humming_or_none",
    "optional_humming",
    "optional_sung_or_instrumental",
    "instrumental_or_short_vocal",
    "vocal_chops_or_sung",
    "breathy_rap",
    "half_spoken_rap",
    "warm_breathy_rap",
}
CARD_ROLES = {"style_card", "composite_query_template"}
RETRIEVAL_SCOPES = {"direct_candidate", "query_expansion_only"}
REVIEW_STATUSES = {"draft", "approved"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any] | None:
    return value if isinstance(value, list) else None


def _content_hash(card: dict[str, Any]) -> str:
    payload = (
        f"{card['summary']}\n{card['prompt_template']}\n"
        f"{','.join(card['backend_tags']['heartmula'])}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_card(card: dict[str, Any], source_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    card_id = str(card.get("id", "<missing-id>"))

    missing = sorted(BASE_REQUIRED_FIELDS - set(card))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    if card.get("kind") != "style_template":
        errors.append("kind_must_be_style_template")
    if card.get("card_role") not in CARD_ROLES:
        errors.append("invalid_card_role")
    if card.get("retrieval_scope") not in RETRIEVAL_SCOPES:
        errors.append("invalid_retrieval_scope")
    if card.get("card_role") == "composite_query_template":
        if card.get("retrieval_scope") != "query_expansion_only":
            errors.append("composite_must_be_query_expansion_only")
        if not card.get("query_expansion_terms"):
            errors.append("composite_requires_query_expansion_terms")
    if card.get("card_role") == "style_card" and card.get("retrieval_scope") != "direct_candidate":
        errors.append("style_card_must_be_direct_candidate")

    for field in ("subgenres", "moods", "reference_bpm_examples", "instruments", "vocal_options", "official_prompt_dimensions", "official_facts", "retrieval_terms", "supported_backends"):
        if _as_list(card.get(field)) is None:
            errors.append(f"{field}_must_be_list")
    if not card.get("retrieval_terms"):
        errors.append("retrieval_terms_must_not_be_empty")
    if not isinstance(card.get("backend_tags"), dict) or not isinstance(card.get("backend_tags", {}).get("heartmula"), list):
        errors.append("backend_tags_heartmula_must_be_list")
    if not isinstance(card.get("backend_validation"), dict):
        errors.append("backend_validation_must_be_object")
    elif card.get("backend_validation", {}).get("heartmula-local") != "unverified":
        errors.append("heartmula_local_must_remain_unverified")

    source_id = str(card.get("source_id", ""))
    if source_id not in source_ids:
        errors.append("source_id_not_in_catalog")
    for field in ("source_title", "source_url", "license_or_usage_note", "derivation_note"):
        if not card.get(field):
            errors.append(f"{field}_must_not_be_empty")

    if card.get("review_status") not in REVIEW_STATUSES:
        errors.append("invalid_review_status")
    if not re.fullmatch(r"[0-9a-f]{64}", str(card.get("content_sha256", ""))):
        errors.append("content_sha256_must_be_64_lower_hex")
    elif _content_hash(card) != card.get("content_sha256"):
        errors.append("content_sha256_mismatch")

    min_bpm = card.get("tempo_bpm_min")
    max_bpm = card.get("tempo_bpm_max")
    if (min_bpm is None) != (max_bpm is None):
        errors.append("tempo_min_max_must_be_both_null_or_numeric")
    if min_bpm is not None and max_bpm is not None:
        if not isinstance(min_bpm, (int, float)) or not isinstance(max_bpm, (int, float)):
            errors.append("tempo_min_max_must_be_numeric")
        elif min_bpm > max_bpm:
            errors.append("tempo_min_must_not_exceed_max")
        if not card.get("tempo_usage_note"):
            warnings.append("numeric_tempo_should_have_tempo_usage_note")
    if any(not isinstance(value, (int, float)) for value in card.get("reference_bpm_examples", [])):
        errors.append("reference_bpm_examples_must_be_numeric")

    for field in ("hidden_retrieval_aliases", "query_expansion_terms", "scene_tags"):
        value = card.get(field)
        if value is not None and not isinstance(value, list):
            errors.append(f"{field}_must_be_list_when_present")
    core = {str(value).strip().lower() for value in card.get("retrieval_terms", [])}
    hidden = {str(value).strip().lower() for value in card.get("hidden_retrieval_aliases", [])}
    expansion = {str(value).strip().lower() for value in card.get("query_expansion_terms", [])}
    scenes = {str(value).strip().lower() for value in card.get("scene_tags", [])}
    if core & scenes:
        errors.append("scene_tags_must_not_duplicate_core_retrieval_terms")
    if core & expansion and card.get("card_role") != "composite_query_template":
        errors.append("query_expansion_terms_must_not_duplicate_core_retrieval_terms")
    if card.get("card_role") == "composite_query_template" and not expansion:
        errors.append("composite_query_expansion_terms_must_not_be_empty")

    present_voice = VOICE_FIELDS & set(card)
    if present_voice and present_voice != VOICE_FIELDS:
        errors.append("voice_fields_must_be_all_present_or_all_absent")
    if present_voice == VOICE_FIELDS:
        if card.get("vocal_type") not in VOICE_TYPES:
            errors.append("invalid_vocal_type")
        if not isinstance(card.get("rap_default"), bool):
            errors.append("rap_default_must_be_bool")
        if not isinstance(card.get("instrumental_default"), bool):
            errors.append("instrumental_default_must_be_bool")
        if not isinstance(card.get("default_language"), str) or not card.get("default_language"):
            errors.append("default_language_must_be_nonempty_string")
        if card.get("rap_default") and card.get("instrumental_default"):
            errors.append("rap_and_instrumental_defaults_cannot_both_be_true")
        if card.get("vocal_type") == "optional_humming" and not card.get("instrumental_default"):
            errors.append("optional_humming_requires_instrumental_default")
    elif card.get("genre") in {"民族国韵", "新世纪 Rap"}:
        warnings.append("voice_fields_missing_on_voice_focused_card")

    return {
        "card_id": card_id,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "review_status": card.get("review_status"),
        "card_role": card.get("card_role"),
        "retrieval_scope": card.get("retrieval_scope"),
        "voice_fields_complete": present_voice == VOICE_FIELDS,
    }


def validate(
    cards: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    schema_implementation_present: bool = False,
) -> dict[str, Any]:
    source_ids = {str(source.get("source_id")) for source in sources}
    card_ids = [str(card.get("id")) for card in cards]
    duplicate_ids = sorted({card_id for card_id in card_ids if card_ids.count(card_id) > 1})
    card_results = [_check_card(card, source_ids) for card in cards]
    errors = [
        {"card_id": row["card_id"], "issues": row["errors"]}
        for row in card_results
        if row["errors"]
    ]
    warnings = [
        {"card_id": row["card_id"], "issues": row["warnings"]}
        for row in card_results
        if row["warnings"]
    ]
    if duplicate_ids:
        errors.append({"card_id": "<dataset>", "issues": ["duplicate_ids:" + ",".join(duplicate_ids)]})
    return {
        "contract_id": "style-template-shadow-contract-v1.2-20260916",
        "contract_mode": "shadow_only",
        "schema_implementation_present": schema_implementation_present,
        "card_count": len(cards),
        "source_count": len(sources),
        "duplicate_card_ids": duplicate_ids,
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "card_results": card_results,
        "review_status_changed": False,
        "production_index_changed": False,
        "interpretation": "valid=true 仅表示当前草稿通过影子字段契约；正式 tools/rag/schemas.py 尚未实现，不能据此宣称生产 RAG 已准入。",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, default=Path("data/miaoxiang/style_templates.v2.jsonl"))
    parser.add_argument("--sources", type=Path, default=Path("data/miaoxiang/source_catalog.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/output/evaluations/style-template-contract-v1-20260916"))
    args = parser.parse_args()
    cards = _read_jsonl(args.cards)
    sources = _read_jsonl(args.sources)
    report = validate(cards, sources, schema_implementation_present=Path("tools/rag/schemas.py").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "contract_id", "contract_mode", "schema_implementation_present", "card_count",
        "source_count", "valid", "error_count", "warning_count",
        "review_status_changed", "production_index_changed",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
