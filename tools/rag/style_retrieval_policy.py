"""Deterministic shadow policy for deciding whether style RAG may be used.

The policy is intentionally conservative: only explicit, high-confidence style
language in the user's raw text can authorize candidates.  The module does not
query Chroma, mutate runtime state, or connect results to the Producer Agent.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tools.rag.schemas import StyleCard
from tools.rag.style_index import load_index_inputs, select_local_demo_cards


Action = Literal["no_rag", "single_style", "multi_candidate"]
Constraint = Literal["required", "forbidden", "optional", "unspecified", "conflict"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ParsedConstraints(_StrictModel):
    rap: Constraint
    instrumental: Constraint
    vocal_gender: Literal["male", "female", "unspecified", "conflict"]
    optional_humming: bool = False


class ExcludedCandidate(_StrictModel):
    card_id: str
    reason: str


class CompositeGuidance(_StrictModel):
    preserve: list[str]
    add: list[str]
    avoid: list[str]


class StyleRetrievalDecision(_StrictModel):
    action: Action
    style_intent_origin: Literal[
        "user_explicit",
        "user_multiple",
        "user_composite",
        "broad_only",
        "low_confidence_only",
        "unsupported_explicit",
        "none",
    ]
    reason: str
    recognized_terms: list[str] = Field(default_factory=list)
    constraints: ParsedConstraints
    candidate_ids: list[str] = Field(default_factory=list, max_length=2)
    excluded_candidates: list[ExcludedCandidate] = Field(default_factory=list)
    shadow_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    composite_template_id: str | None = None
    composite_guidance: CompositeGuidance | None = None
    runtime_injection_allowed: Literal[False] = False


# Only these terms can authorize a direct candidate.  Instrument, mood, scene,
# and hidden-alias terms stay out of this registry unless they uniquely name a
# reviewed style in the current project.
HIGH_CONFIDENCE_TERMS: tuple[tuple[str, str], ...] = (
    ("deep house", "style-edm-deep-house-v2"),
    ("深度浩室", "style-edm-deep-house-v2"),
    ("深浩室", "style-edm-deep-house-v2"),
    ("tropical house", "style-edm-tropical-house-v2"),
    ("热带浩室", "style-edm-tropical-house-v2"),
    ("vina house", "style-dynamic-vina-house-v2"),
    ("越南鼓", "style-dynamic-vina-house-v2"),
    ("chinese bounce", "style-dynamic-chinese-bounce-v2"),
    ("sad slap house", "style-dynamic-sad-slap-house-v2"),
    ("slap house", "style-dynamic-sad-slap-house-v2"),
    ("伤感舞曲", "style-dynamic-sad-slap-house-v2"),
    ("苦情电音", "style-dynamic-sad-slap-house-v2"),
    ("techno", "style-edm-techno-v2"),
    ("future bass", "style-edm-future-bass-v2"),
    ("未来贝斯", "style-edm-future-bass-v2"),
    ("trance", "style-edm-trance-v2"),
    ("迷幻舞曲", "style-edm-trance-v2"),
    ("drum & bass", "style-edm-drum-and-bass-v2"),
    ("drum and bass", "style-edm-drum-and-bass-v2"),
    ("dnb", "style-edm-drum-and-bass-v2"),
    ("唯美抒情国风", "style-guofeng-elegant-ballad-v2"),
    ("唯美国风", "style-guofeng-elegant-ballad-v2"),
    ("古风抒情", "style-guofeng-elegant-ballad-v2"),
    ("宏大武侠", "style-guofeng-epic-wuxia-v2"),
    ("电影感国风", "style-guofeng-epic-wuxia-v2"),
    ("仙侠", "style-guofeng-epic-wuxia-v2"),
    ("禅意国风", "style-guofeng-zen-ambient-v2"),
    ("新国风电子", "style-guofeng-electronic-fusion-v2"),
    ("国风电子", "style-guofeng-electronic-fusion-v2"),
    ("赛博东方", "style-guofeng-electronic-fusion-v2"),
    ("zen fusion rap", "style-rap-zen-fusion-v2"),
    ("禅意说唱", "style-rap-zen-fusion-v2"),
    ("禅意中文说唱", "style-rap-zen-fusion-v2"),
    ("cyber zen rap", "style-rap-cyber-zen-v2"),
    ("赛博禅意", "style-rap-cyber-zen-v2"),
    ("灵动疗愈系 rap", "style-rap-healing-new-age-v2"),
    ("疗愈说唱", "style-rap-healing-new-age-v2"),
    ("new age rap", "style-rap-healing-new-age-v2"),
)

BROAD_TERMS = (
    "house",
    "浩室",
    "edm",
    "电子",
    "电音",
    "国风",
    "中国风",
    "古风",
    "rap",
    "说唱",
    "舞曲",
    "民谣",
)

LOW_CONFIDENCE_ALIASES: dict[str, str] = {
    "慢摇": "style-dynamic-chinese-bounce-v2",
    "车载dj": "style-dynamic-vina-house-v2",
    "土嗨": "style-dynamic-vina-house-v2",
    "emotional remix": "style-dynamic-sad-slap-house-v2",
}

LOW_ALIAS_CORROBORATION: dict[str, tuple[str, ...]] = {
    "慢摇": ("弹性贝斯", "四拍", "bounce"),
    "车载dj": ("越南鼓", "反拍贝斯", "烟嗓"),
    "土嗨": ("越南鼓", "反拍贝斯", "烟嗓"),
    "emotional remix": ("悲伤钢琴", "bass pulse", "slap house"),
}

UNSUPPORTED_STYLE_TERMS = (
    "jazz",
    "爵士",
    "bossa nova",
    "波萨诺瓦",
    "punk",
    "朋克",
    "rock",
    "摇滚",
    "metal",
    "金属",
    "reggae",
    "雷鬼",
    "city pop",
)

COMPOSITE_POLICIES: tuple[dict[str, Any], ...] = (
    {
        "terms": ("苦情民谣电音", "江湖沧桑民谣电音"),
        "template_id": "style-dynamic-sad-folk-pop-remix-v2",
        "candidate_ids": (
            "style-dynamic-sad-slap-house-v2",
            "style-dynamic-vina-house-v2",
        ),
        "guidance": {
            "preserve": ["民谣叙事", "烟嗓质感", "夜路或江湖氛围"],
            "add": ["克制电子鼓", "低频脉冲"],
            "avoid": ["堆叠多个 Drop", "把民谣完全改成夜店舞曲"],
        },
    },
)

RAP_COMPANIONS = {
    "style-guofeng-electronic-fusion-v2": "style-rap-cyber-zen-v2",
    "style-guofeng-zen-ambient-v2": "style-rap-zen-fusion-v2",
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _contains(text: str, term: str) -> bool:
    term = _normalize(term)
    if re.fullmatch(r"[a-z0-9 &+_-]+", term):
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None
    return term in text


def _first_index(text: str, term: str) -> int:
    normalized_term = _normalize(term)
    if re.fullmatch(r"[a-z0-9 &+_-]+", normalized_term):
        escaped = re.escape(normalized_term).replace(r"\ ", r"\s+")
        match = re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text)
        return match.start() if match else len(text) + 1
    index = text.find(normalized_term)
    return index if index >= 0 else len(text) + 1


def _has_ascii_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) is not None


def parse_constraints(user_text: str) -> ParsedConstraints:
    text = _normalize(user_text)
    optional_humming = "哼鸣" in text

    rap_optional = any(
        phrase in text
        for phrase in ("说唱可有可无", "rap可有可无", "不一定要说唱", "可以有一点rap", "主歌可以念白")
    )
    rap_forbidden = any(
        phrase in text
        for phrase in ("不要说唱", "不要rap", "不说唱", "非说唱", "全程演唱")
    )
    rap_required = any(
        phrase in text for phrase in ("说唱", "半说半唱", "气声说唱", "念白式")
    ) or _has_ascii_word(text, "rap")
    if rap_optional:
        rap: Constraint = "optional"
    elif rap_forbidden:
        rap = "forbidden"
    elif rap_required:
        rap = "required"
    else:
        rap = "unspecified"

    instrumental_required = any(
        phrase in text for phrase in ("纯音乐", "无人声", "不要唱", "instrumental")
    )
    vocal_forbidden = any(
        phrase in text for phrase in ("不要正式演唱", "不要演唱", "不要人声", "无人声")
    )
    vocal_required = not vocal_forbidden and (
        any(phrase in text for phrase in ("主唱", "对唱", "演唱", "有人声"))
        or (not optional_humming and any(phrase in text for phrase in ("女声", "男声")))
    )
    if instrumental_required and vocal_required:
        instrumental: Constraint = "conflict"
    elif instrumental_required:
        instrumental = "required"
    elif vocal_required:
        instrumental = "forbidden"
    else:
        instrumental = "unspecified"

    male = "男声" in text
    female = "女声" in text
    if male and female:
        gender: Literal["male", "female", "unspecified", "conflict"] = "conflict"
    elif male:
        gender = "male"
    elif female:
        gender = "female"
    else:
        gender = "unspecified"
    return ParsedConstraints(
        rap=rap,
        instrumental=instrumental,
        vocal_gender=gender,
        optional_humming=optional_humming,
    )


def _direct_cards() -> tuple[dict[str, StyleCard], set[str]]:
    cards, sources, policy = load_index_inputs()
    direct, composite, blocked = select_local_demo_cards(cards, sources, policy)
    if blocked:
        raise RuntimeError(f"current_admission_state_is_blocked:{blocked}")
    return {card.id: card for card in direct}, {card.id for card in composite}


def _explicit_matches(text: str) -> list[tuple[int, str, str]]:
    matches: list[tuple[int, str, str]] = []
    seen_cards: set[str] = set()
    for term, card_id in HIGH_CONFIDENCE_TERMS:
        if _contains(text, term) and card_id not in seen_cards:
            matches.append((_first_index(text, term), term, card_id))
            seen_cards.add(card_id)
    for alias, card_id in LOW_CONFIDENCE_ALIASES.items():
        if card_id in seen_cards or not _contains(text, alias):
            continue
        if any(_contains(text, evidence) for evidence in LOW_ALIAS_CORROBORATION[alias]):
            matches.append((_first_index(text, alias), alias, card_id))
            seen_cards.add(card_id)
    return sorted(matches, key=lambda row: (row[0], row[1]))


def _incompatibility(card: StyleCard, constraints: ParsedConstraints) -> str | None:
    if constraints.rap == "required" and card.rap_default is not True:
        return "rap_required_but_card_not_rap_default"
    if constraints.rap == "forbidden" and card.rap_default is True:
        return "rap_forbidden_but_card_is_rap_default"
    if constraints.instrumental == "required" and card.instrumental_default is False:
        return "instrumental_required_but_card_not_instrumental_default"
    if constraints.instrumental == "forbidden" and card.instrumental_default is True:
        if not (constraints.optional_humming and card.vocal_type == "optional_humming"):
            return "vocal_required_but_card_instrumental_default"
    if constraints.instrumental == "conflict":
        return "conflicting_instrumental_and_vocal_request"
    return None


def _no_rag(
    *,
    origin: str,
    reason: str,
    constraints: ParsedConstraints,
    recognized_terms: list[str] | None = None,
    excluded: list[ExcludedCandidate] | None = None,
) -> StyleRetrievalDecision:
    return StyleRetrievalDecision(
        action="no_rag",
        style_intent_origin=origin,
        reason=reason,
        recognized_terms=recognized_terms or [],
        constraints=constraints,
        excluded_candidates=excluded or [],
        runtime_injection_allowed=False,
    )


def decide_style_retrieval(user_text: str) -> StyleRetrievalDecision:
    """Return a fail-closed shadow decision from raw user text only."""

    if not user_text.strip():
        raise ValueError("user_text_must_not_be_empty")
    text = _normalize(user_text)
    constraints = parse_constraints(text)
    direct_by_id, composite_ids = _direct_cards()

    for policy in COMPOSITE_POLICIES:
        matched_terms = [term for term in policy["terms"] if _contains(text, term)]
        if not matched_terms:
            continue
        template_id = policy["template_id"]
        if template_id not in composite_ids:
            raise RuntimeError(f"composite_template_not_admitted:{template_id}")
        candidate_ids = list(policy["candidate_ids"])
        missing = [card_id for card_id in candidate_ids if card_id not in direct_by_id]
        if missing:
            raise RuntimeError(f"composite_candidate_not_admitted:{missing}")
        excluded: list[ExcludedCandidate] = []
        compatible: list[str] = []
        for card_id in candidate_ids:
            reason = _incompatibility(direct_by_id[card_id], constraints)
            if reason:
                excluded.append(ExcludedCandidate(card_id=card_id, reason=reason))
            else:
                compatible.append(card_id)
        if not compatible:
            return _no_rag(
                origin="user_composite",
                reason="composite_candidates_conflict_with_hard_constraints",
                constraints=constraints,
                recognized_terms=matched_terms,
                excluded=excluded,
            )
        return StyleRetrievalDecision(
            action="single_style" if len(compatible) == 1 else "multi_candidate",
            style_intent_origin="user_composite",
            reason="named_composite",
            recognized_terms=matched_terms,
            constraints=constraints,
            candidate_ids=compatible[:2],
            excluded_candidates=excluded,
            composite_template_id=template_id,
            composite_guidance=CompositeGuidance.model_validate(policy["guidance"]),
            runtime_injection_allowed=False,
        )

    matches = _explicit_matches(text)
    recognized_terms = [term for _, term, _ in matches]
    candidate_ids = [card_id for _, _, card_id in matches]
    unsupported = [term for term in UNSUPPORTED_STYLE_TERMS if _contains(text, term)]
    if candidate_ids and unsupported:
        return _no_rag(
            origin="unsupported_explicit",
            reason="partially_unsupported_multiple_style",
            constraints=constraints,
            recognized_terms=recognized_terms + unsupported,
        )

    # Optional or required rap may request a reviewed rap-side alternative for
    # a specifically paired non-rap card.  This is a transparent project rule,
    # not a vector-score inference.
    if constraints.rap in {"optional", "required"}:
        for card_id in list(candidate_ids):
            companion = RAP_COMPANIONS.get(card_id)
            if companion and companion not in candidate_ids:
                candidate_ids.append(companion)

    excluded: list[ExcludedCandidate] = []
    compatible: list[str] = []
    for card_id in candidate_ids:
        card = direct_by_id.get(card_id)
        if card is None:
            raise RuntimeError(f"explicit_style_card_not_admitted:{card_id}")
        reason = _incompatibility(card, constraints)
        if reason:
            excluded.append(ExcludedCandidate(card_id=card_id, reason=reason))
        else:
            compatible.append(card_id)

    if candidate_ids and not compatible:
        return _no_rag(
            origin="user_explicit" if len(candidate_ids) == 1 else "user_multiple",
            reason="explicit_style_conflicts_with_hard_constraints",
            constraints=constraints,
            recognized_terms=recognized_terms,
            excluded=excluded,
        )
    if compatible:
        compatible = compatible[:2]
        return StyleRetrievalDecision(
            action="single_style" if len(compatible) == 1 else "multi_candidate",
            style_intent_origin=(
                "user_explicit" if len(compatible) == 1 else "user_multiple"
            ),
            reason=(
                "explicit_supported_style"
                if len(compatible) == 1
                else "explicit_multiple_styles"
            ),
            recognized_terms=recognized_terms,
            constraints=constraints,
            candidate_ids=compatible,
            excluded_candidates=excluded,
            runtime_injection_allowed=False,
        )

    low_terms = [term for term in LOW_CONFIDENCE_ALIASES if _contains(text, term)]
    broad_terms = [term for term in BROAD_TERMS if _contains(text, term)]
    if low_terms:
        return _no_rag(
            origin="low_confidence_only",
            reason="low_confidence_alias_without_corroboration",
            constraints=constraints,
            recognized_terms=low_terms,
        )
    if unsupported:
        return _no_rag(
            origin="unsupported_explicit",
            reason="unsupported_explicit_style",
            constraints=constraints,
            recognized_terms=unsupported,
        )
    if broad_terms:
        return _no_rag(
            origin="broad_only",
            reason="broad_style_only",
            constraints=constraints,
            recognized_terms=broad_terms,
        )
    return _no_rag(
        origin="none",
        reason="no_explicit_supported_style",
        constraints=constraints,
    )
