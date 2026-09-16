"""Formal data contracts for the draft style-card RAG dataset.

This module validates records and exposes the production-admission predicate,
but it does not build or mutate a Chroma collection.  Admission is deliberately
separate from parsing: a valid draft card can still be ineligible for the
production index.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceRecord(_StrictModel):
    source_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    public_access_label: str = Field(min_length=1)
    access_type: str = Field(min_length=1)
    accessed_at: str = Field(min_length=1)
    last_modified_display: str | None = None
    license_status: str = Field(min_length=1)
    allowed_project_use: str = Field(min_length=1)
    raw_content_saved: bool
    source_content_sha256: str | None = None
    verification_status: str = Field(min_length=1)
    notes: str = Field(min_length=1)


class BackendValidation(_StrictModel):
    heartmula_local: str = Field(alias="heartmula-local", min_length=1)

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class SupplementalSource(_StrictModel):
    source_id: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license_status: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    changes: str = Field(min_length=1)


SourceUseScope = Literal["local_demo", "public_product"]


SourceAdmissionStatus = Literal[
    "evaluation_only",
    "local_demo_allowed",
    "public_product_allowed",
    "blocked",
]


class SourceAdmissionDecision(_StrictModel):
    status: SourceAdmissionStatus
    reason: str = Field(min_length=1)


class SourceAdmissionPolicy(_StrictModel):
    schema_version: str = Field(min_length=1)
    default_status: SourceAdmissionStatus
    policy_note: str = Field(min_length=1)
    sources: dict[str, SourceAdmissionDecision] = Field(default_factory=dict)
    cards: dict[str, SourceAdmissionDecision] = Field(default_factory=dict)


class StyleCard(_StrictModel):
    id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    kind: Literal["style_template"]
    genre: str = Field(min_length=1)
    subgenres: list[str]
    moods: list[str]
    tempo_bpm_min: int | float | None = None
    tempo_bpm_max: int | float | None = None
    reference_bpm_examples: list[int | float]
    instruments: list[str]
    vocal_options: list[str]
    melody: str = Field(min_length=1)
    dynamics: str = Field(min_length=1)
    structure_notes: str = Field(min_length=1)
    official_prompt_dimensions: list[str]
    official_facts: list[str]
    evidence_depth: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    prompt_template: str = Field(min_length=1)
    retrieval_terms: list[str]
    hidden_retrieval_aliases: list[str] = Field(default_factory=list)
    query_expansion_terms: list[str] = Field(default_factory=list)
    scene_tags: list[str] = Field(default_factory=list)
    backend_tags: dict[str, list[str]]
    backend_tag_source: str = Field(min_length=1)
    supported_backends: list[str]
    backend_validation: BackendValidation
    supplemental_sources: list[SupplementalSource] = Field(default_factory=list)
    supplemental_facts: list[str] = Field(default_factory=list)
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    accessed_at: str = Field(min_length=1)
    license_or_usage_note: str = Field(min_length=1)
    derivation_note: str = Field(min_length=1)
    content_sha256: str = Field(min_length=1)
    content_hash_scope: str = Field(min_length=1)
    review_status: Literal["draft", "approved"]
    version: int
    card_role: Literal["style_card", "composite_query_template"]
    retrieval_scope: Literal["direct_candidate", "query_expansion_only"]
    tempo_usage_note: str | None = None
    admission_note: str | None = None
    vocal_type: str | None = None
    rap_default: bool | None = None
    instrumental_default: bool | None = None
    default_language: str | None = None
    retrieval_policy_notes: str | None = None

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> "StyleCard":
        if self.card_role == "style_card" and self.retrieval_scope != "direct_candidate":
            raise ValueError("style_card_must_be_direct_candidate")
        if self.card_role == "composite_query_template":
            if self.retrieval_scope != "query_expansion_only":
                raise ValueError("composite_must_be_query_expansion_only")
            if not self.query_expansion_terms:
                raise ValueError("composite_requires_query_expansion_terms")
        if not self.retrieval_terms:
            raise ValueError("retrieval_terms_must_not_be_empty")
        if (self.tempo_bpm_min is None) != (self.tempo_bpm_max is None):
            raise ValueError("tempo_min_max_must_be_both_null_or_numeric")
        if self.tempo_bpm_min is not None and self.tempo_bpm_max is not None:
            if self.tempo_bpm_min > self.tempo_bpm_max:
                raise ValueError("tempo_min_must_not_exceed_max")
        voice_values = (
            self.vocal_type,
            self.rap_default,
            self.instrumental_default,
            self.default_language,
        )
        if any(value is not None for value in voice_values) and not all(
            value is not None for value in voice_values
        ):
            raise ValueError("voice_fields_must_be_all_present_or_all_absent")
        if self.rap_default and self.instrumental_default:
            raise ValueError("rap_and_instrumental_defaults_cannot_both_be_true")
        if self.vocal_type == "optional_humming" and not self.instrumental_default:
            raise ValueError("optional_humming_requires_instrumental_default")
        core_terms = {term.strip().lower() for term in self.retrieval_terms}
        if core_terms & {term.strip().lower() for term in self.scene_tags}:
            raise ValueError("scene_tags_must_not_duplicate_core_retrieval_terms")
        if self.card_role != "composite_query_template" and core_terms & {
            term.strip().lower() for term in self.query_expansion_terms
        }:
            raise ValueError("query_expansion_terms_must_not_duplicate_core_retrieval_terms")
        return self


def compute_style_card_hash(card: StyleCard) -> str:
    """Recompute the content hash used by the current v2 card records."""

    payload = (
        f"{card.summary}\n{card.prompt_template}\n"
        f"{','.join(card.backend_tags.get('heartmula', []))}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_style_card_hash(card: StyleCard) -> bool:
    """Return whether the stored hash matches the contract payload."""

    return compute_style_card_hash(card) == card.content_sha256


def source_allows_derivative_index(source: SourceRecord) -> tuple[bool, str | None]:
    """Check the current project's source-use boundary for derived cards."""

    if source.verification_status != "verified_readable":
        return False, "source_not_verified_readable"
    if not source.allowed_project_use:
        return False, "source_allowed_project_use_missing"
    if source.license_status == "no_redistribution_license_observed" and source.raw_content_saved:
        return False, "restricted_source_raw_content_saved"
    return True, None


def production_admission(
    card: StyleCard,
    source: SourceRecord | None,
    *,
    source_admission_statuses: list[SourceAdmissionStatus] | None = None,
    card_admission_status: SourceAdmissionStatus = "evaluation_only",
    requested_scope: SourceUseScope = "public_product",
    contract_errors: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Return a scope-specific index decision without mutating either record."""

    reasons = list(contract_errors or [])
    if card.review_status != "approved":
        reasons.append("review_status_not_approved")
    if card.card_role != "style_card" or card.retrieval_scope != "direct_candidate":
        reasons.append("not_a_direct_style_candidate")
    def status_allows_scope(status: SourceAdmissionStatus) -> bool:
        if requested_scope == "local_demo":
            return status in {"local_demo_allowed", "public_product_allowed"}
        return status == "public_product_allowed"

    effective_source_statuses = source_admission_statuses or ["evaluation_only"]
    if not all(status_allows_scope(status) for status in effective_source_statuses):
        reasons.append(f"source_admission_not_granted_for_{requested_scope}")
    if not status_allows_scope(card_admission_status):
        reasons.append(f"card_admission_not_granted_for_{requested_scope}")
    if source is None:
        reasons.append("source_not_resolved")
    else:
        source_ok, source_reason = source_allows_derivative_index(source)
        if not source_ok and source_reason:
            reasons.append(source_reason)
    if not validate_style_card_hash(card):
        reasons.append("content_sha256_mismatch")
    return not reasons, reasons


class StyleQuery(_StrictModel):
    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    keywords: list[str]
    primary_card_ids: list[str]
    expected_card_ids: list[str]
    expected_scope_note: str | None = None

    @model_validator(mode="after")
    def validate_query_labels(self) -> "StyleQuery":
        if not self.keywords:
            raise ValueError("query_keywords_must_not_be_empty")
        if not self.primary_card_ids:
            raise ValueError("query_primary_card_ids_must_not_be_empty")
        if not set(self.primary_card_ids).issubset(self.expected_card_ids):
            raise ValueError("query_primary_must_be_in_expected_cards")
        return self


class StyleQueryManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["style_fixed_query_set"]
    query_set_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    retrieval_mode: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    parent_manifest: str | None = None
    cards_path: str = Field(min_length=1)
    source_catalog_path: str = Field(min_length=1)
    query_scope_note: str | None = None
    lexical_result_path: str | None = None
    queries: list[StyleQuery]

    @model_validator(mode="after")
    def validate_query_ids(self) -> "StyleQueryManifest":
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_ids_must_be_unique")
        if not self.queries:
            raise ValueError("queries_must_not_be_empty")
        return self
