"""Selection and document helpers for the local-demo style index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.rag.schemas import (
    SourceAdmissionPolicy,
    SourceRecord,
    StyleCard,
    production_admission,
)


COLLECTION_NAME = "style_templates_local_demo_v1"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_CACHE = Path("tmp/hf-cache/hub")
INDEX_PATH = Path("data/rag/chroma/style_templates_local_demo_v1")
CARDS_PATH = Path("data/miaoxiang/style_templates.v2.jsonl")
SOURCES_PATH = Path("data/miaoxiang/source_catalog.jsonl")
ADMISSION_AUDIT_PATH = Path("data/miaoxiang/style_templates.v2.admission-audit.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_index_inputs(
    cards_path: Path = CARDS_PATH,
    sources_path: Path = SOURCES_PATH,
    admission_audit_path: Path = ADMISSION_AUDIT_PATH,
) -> tuple[list[StyleCard], dict[str, SourceRecord], SourceAdmissionPolicy]:
    cards = [StyleCard.model_validate(row) for row in read_jsonl(cards_path)]
    sources = [SourceRecord.model_validate(row) for row in read_jsonl(sources_path)]
    audit = json.loads(admission_audit_path.read_text(encoding="utf-8"))
    policy = SourceAdmissionPolicy.model_validate(audit["source_use_admission"])
    return cards, {source.source_id: source for source in sources}, policy


def _status_for_source(policy: SourceAdmissionPolicy, source_id: str) -> str:
    decision = policy.sources.get(source_id)
    return decision.status if decision else policy.default_status


def select_local_demo_cards(
    cards: list[StyleCard],
    sources: dict[str, SourceRecord],
    policy: SourceAdmissionPolicy,
) -> tuple[list[StyleCard], list[StyleCard], dict[str, list[str]]]:
    """Return direct cards, composite templates, and blocked reasons."""

    direct: list[StyleCard] = []
    composite: list[StyleCard] = []
    blocked: dict[str, list[str]] = {}
    for card in cards:
        source_ids = [card.source_id] + [row.source_id for row in card.supplemental_sources]
        missing_source_ids = [source_id for source_id in source_ids if source_id not in sources]
        contract_errors = [f"source_not_resolved:{source_id}" for source_id in missing_source_ids]
        source_statuses = [_status_for_source(policy, source_id) for source_id in source_ids]
        card_decision = policy.cards.get(card.id)
        card_status = card_decision.status if card_decision else policy.default_status
        eligible, reasons = production_admission(
            card,
            sources.get(card.source_id),
            source_admission_statuses=source_statuses,
            card_admission_status=card_status,
            requested_scope="local_demo",
            contract_errors=contract_errors,
        )
        if card.card_role == "composite_query_template":
            composite.append(card)
            non_role_reasons = [
                reason for reason in reasons if reason != "not_a_direct_style_candidate"
            ]
            if non_role_reasons:
                blocked[card.id] = non_role_reasons
        elif eligible:
            direct.append(card)
        else:
            blocked[card.id] = reasons
    return direct, composite, blocked


def build_embedding_document(card: StyleCard) -> str:
    """Build retrieval text without generation, licensing, or backend-control fields."""

    parts = [
        f"风格：{card.genre}",
        f"子风格：{'、'.join(card.subgenres)}",
        f"摘要：{card.summary}",
        f"核心检索词：{'、'.join(card.retrieval_terms)}",
    ]
    if card.hidden_retrieval_aliases:
        parts.append(f"中文别名：{'、'.join(card.hidden_retrieval_aliases)}")
    if card.moods:
        parts.append(f"情绪：{'、'.join(card.moods)}")
    if card.instruments:
        parts.append(f"音色与乐器：{'、'.join(card.instruments)}")
    if card.vocal_type:
        parts.append(f"人声类型：{card.vocal_type}")
        parts.append(f"默认说唱：{'是' if card.rap_default else '否'}")
        parts.append(f"默认纯音乐：{'是' if card.instrumental_default else '否'}")
        parts.append(f"默认语言：{card.default_language}")
    return "。".join(parts) + "。"


def build_metadata(card: StyleCard) -> dict[str, str | bool | int | float]:
    return {
        "card_id": card.id,
        "content_sha256": card.content_sha256,
        "schema_version": card.schema_version,
        "genre": card.genre,
        "subgenres_json": json.dumps(card.subgenres, ensure_ascii=False),
        "review_status": card.review_status,
        "card_role": card.card_role,
        "retrieval_scope": card.retrieval_scope,
        "source_scope": "local_demo_allowed",
        "primary_source_id": card.source_id,
        "supplemental_source_ids_json": json.dumps(
            [row.source_id for row in card.supplemental_sources],
            ensure_ascii=False,
        ),
        "source_title": card.source_title,
        "source_url": card.source_url,
        "heartmula_validation": card.backend_validation.heartmula_local,
    }


def expand_composite_query(
    query: str,
    composite_cards: list[StyleCard],
) -> tuple[str, list[str]]:
    """Append expansion terms only when a named composite expression is present."""

    normalized = "".join(query.lower().split())
    matched: list[StyleCard] = []
    for card in composite_cards:
        triggers = list(card.retrieval_terms) + list(card.hidden_retrieval_aliases)
        if any("".join(term.lower().split()) in normalized for term in triggers):
            matched.append(card)
    if not matched:
        return query, []
    expansion_terms: list[str] = []
    for card in matched:
        for term in card.query_expansion_terms:
            if term not in expansion_terms:
                expansion_terms.append(term)
    expanded = f"{query} {' '.join(expansion_terms)}".strip()
    return expanded, [card.id for card in matched]
