"""Prepare a per-card admission review packet for the style-card draft set.

This tool is deliberately advisory.  It joins card fields, source evidence,
prior AI/audit decisions, and observed retrieval neighbors into a human review
packet.  It never changes ``review_status`` and never creates or updates a
production index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.prepare_style_near_neighbor_review import GROUPS


DECISION_OPTIONS = ["keep", "revise", "hold_for_source"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _review_by_card(document: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not document:
        return {}
    return {
        str(row["card_id"]): row
        for row in document.get("card_reviews", [])
        if row.get("card_id")
    }


def _group_by_card() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        for card_id in group["card_ids"]:
            result[card_id] = {
                "group_id": group["group_id"],
                "group_label": group["label"],
                "review_questions": list(group["review_questions"]),
            }
    return result


def _source_summary(source: dict[str, Any] | None) -> dict[str, Any] | None:
    if source is None:
        return None
    return {
        "source_id": source.get("source_id"),
        "source_type": source.get("source_type"),
        "publisher": source.get("publisher"),
        "title": source.get("title"),
        "url": source.get("url"),
        "license_status": source.get("license_status"),
        "verification_status": source.get("verification_status"),
        "allowed_project_use": source.get("allowed_project_use"),
        "notes": source.get("notes"),
    }


def _embedding_observations(
    card_id: str, embedding_result: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not embedding_result:
        return []
    rows: list[dict[str, Any]] = []
    embedding = embedding_result.get("embedding", {})
    for result in embedding.get("results", []):
        for rank, neighbor in enumerate(result.get("top3", []), start=1):
            if neighbor.get("card_id") != card_id:
                continue
            rows.append(
                {
                    "query_id": result.get("query_id"),
                    "query_text": result.get("query_text"),
                    "rank": rank,
                    "cosine_distance": neighbor.get("cosine_distance"),
                    "primary_card_ids": result.get("primary_card_ids", []),
                    "top1_primary_hit": result.get("top1_primary_hit"),
                    "top3_expected_hit": result.get("top3_expected_hit"),
                    "top3_source_traceable": result.get("top3_source_traceable"),
                }
            )
    return rows


def _policy_observations(
    card_id: str, policy_result: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not policy_result:
        return []
    rows: list[dict[str, Any]] = []
    for result in policy_result.get("results", []):
        direct_rank = next(
            (
                rank
                for rank, row in enumerate(result.get("direct_top3", []), start=1)
                if row.get("card_id") == card_id
            ),
            None,
        )
        template_top = result.get("template_top") or {}
        if direct_rank is None and template_top.get("card_id") != card_id:
            continue
        rows.append(
            {
                "query_id": result.get("query_id"),
                "query_text": result.get("query_text"),
                "policy": result.get("policy", {}),
                "template_top": template_top.get("card_id"),
                "direct_rank_in_top3": direct_rank,
                "direct_primary_hit": result.get("direct_primary_hit"),
                "direct_excludes_composite": result.get("direct_excludes_composite"),
            }
        )
    return rows


def _card_review_questions(card: dict[str, Any], group: dict[str, Any]) -> list[str]:
    questions = list(group.get("review_questions", []))
    if card.get("card_role") == "composite_query_template":
        questions.insert(
            0,
            "该卡是否应继续只作为组合查询扩展，不作为直接风格候选？",
        )
    else:
        questions.insert(
            0,
            "该卡是否表达一个边界清晰、可独立检索的风格，而不是过宽的项目组合？",
        )
    if any(
        key in card
        for key in ("vocal_type", "rap_default", "instrumental_default", "default_language")
    ):
        questions.append("人声、说唱、纯器乐和中文语种属性是否与摘要及检索词一致？")
    questions.append("BPM、音色和 HeartMuLa 标签是否被清楚标记为参考/候选，而非已验证控制能力？")
    return questions


def prepare(
    cards: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    embedding_result: dict[str, Any] | None = None,
    policy_result: dict[str, Any] | None = None,
    ai_audit: dict[str, Any] | None = None,
    admission_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_by_id = {str(source["source_id"]): source for source in sources}
    group_by_card = _group_by_card()
    ai_by_card = _review_by_card(ai_audit)
    admission_by_card = _review_by_card(admission_audit)
    human_content_by_card = {
        str(row["card_id"]): row
        for row in (admission_audit or {}).get("human_card_review", {}).get("cards", [])
        if row.get("card_id")
    }
    cards_out: list[dict[str, Any]] = []
    missing_sources: list[str] = []
    ungrouped_cards: list[str] = []

    for card in cards:
        card_id = str(card["id"])
        group = group_by_card.get(card_id)
        if group is None:
            ungrouped_cards.append(card_id)
            group = {
                "group_id": None,
                "group_label": None,
                "review_questions": [],
            }
        source = source_by_id.get(str(card.get("source_id")))
        if source is None:
            missing_sources.append(card_id)
        human_content = human_content_by_card.get(card_id, {})
        cards_out.append(
            {
                "card_id": card_id,
                "display_name": card.get("subgenres", [card_id])[0],
                "genre": card.get("genre"),
                "subgenres": card.get("subgenres", []),
                "moods": card.get("moods", []),
                "summary": card.get("summary"),
                "card_role": card.get("card_role", "style_card"),
                "retrieval_scope": card.get("retrieval_scope", "direct_candidate"),
                "direct_candidate": card.get("retrieval_scope", "direct_candidate")
                == "direct_candidate",
                "retrieval_terms": card.get("retrieval_terms", []),
                "hidden_retrieval_aliases": card.get("hidden_retrieval_aliases", []),
                "query_expansion_terms": card.get("query_expansion_terms", []),
                "scene_tags": card.get("scene_tags", []),
                "admission_note": card.get("admission_note"),
                "tempo": {
                    "min_bpm": card.get("tempo_bpm_min"),
                    "max_bpm": card.get("tempo_bpm_max"),
                    "reference_examples": card.get("reference_bpm_examples", []),
                    "usage_note": card.get("tempo_usage_note"),
                },
                "instruments": card.get("instruments", []),
                "vocal_options": card.get("vocal_options", []),
                "vocal_type": card.get("vocal_type"),
                "rap_default": card.get("rap_default"),
                "instrumental_default": card.get("instrumental_default"),
                "default_language": card.get("default_language"),
                "retrieval_policy_notes": card.get("retrieval_policy_notes", []),
                "backend_tags": card.get("backend_tags", {}),
                "backend_validation": card.get("backend_validation", {}),
                "source": _source_summary(source),
                "evidence_depth": card.get("evidence_depth"),
                "content_sha256": card.get("content_sha256"),
                "review_status": card.get("review_status"),
                "version": card.get("version"),
                "group": group,
                "prior_ai_review": ai_by_card.get(card_id),
                "prior_admission_review": admission_by_card.get(card_id),
                "embedding_observations": _embedding_observations(card_id, embedding_result),
                "policy_observations": _policy_observations(card_id, policy_result),
                "review_questions": _card_review_questions(card, group),
                "human_decision": human_content.get("decision"),
                "human_notes": human_content.get("notes"),
                "decision_source": "user_manual_review" if human_content else None,
            }
        )

    card_ids = [card["card_id"] for card in cards_out]
    return {
        "packet_id": "style-card-admission-review-v1-20260916",
        "packet_type": "per_card_human_review_recorded" if human_content_by_card else "per_card_human_review_pending",
        "purpose": "在生产索引前逐卡确认来源、语义边界、检索字段和人声属性。",
        "card_count": len(cards_out),
        "group_count": len({card["group"]["group_id"] for card in cards_out}),
        "card_ids_unique": len(card_ids) == len(set(card_ids)),
        "missing_source_card_ids": missing_sources,
        "ungrouped_card_ids": ungrouped_cards,
        "review_status_mutated": False,
        "production_index_changed": False,
        "human_review_recorded": bool(human_content_by_card),
        "decision_options": DECISION_OPTIONS,
        "group_context": [
            {
                "group_id": group["group_id"],
                "label": group["label"],
                "card_ids": group["card_ids"],
            }
            for group in GROUPS
        ],
        "cards": cards_out,
        "instructions": [
            "逐卡判断，不因固定查询或 embedding top-1 命中就批准。",
            "keep 表示当前内容和边界可保留；revise 表示需要修改字段后重新检查；hold_for_source 表示来源不足。",
            "组合查询模板不得作为 direct_candidate；隐藏别名只用于召回，不展示给用户或直接注入生成提示。",
            "人工决策若已在准入审计中登记，会在本包中镜像显示；本包不写回卡片 JSONL，也不把 keep 解释为 approved。",
        ],
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# 风格卡逐卡准入审核包",
        "",
        f"- packet_id: `{packet['packet_id']}`",
        f"- 卡片数: {packet['card_count']}",
        f"- 分组数: {packet['group_count']}",
        "- 当前状态: 所有人工决策为空；不改变 `review_status`，不创建生产索引",
        "",
        "## 使用说明",
        "",
        "逐卡填写 `keep`、`revise` 或 `hold_for_source`。不要因 top-1 命中直接批准。组合卡只能作为查询扩展，不能作为直接候选。",
        "",
    ]
    for card in packet["cards"]:
        source = card.get("source") or {}
        group = card.get("group") or {}
        lines.extend(
            [
                f"## {card['card_id']}｜{card.get('display_name')}",
                "",
                f"- 分组：{group.get('group_label') or '未分组'}",
                f"- 角色：`{card.get('card_role')}`；检索范围：`{card.get('retrieval_scope')}`",
                f"- 当前状态：`{card.get('review_status')}`；直接候选：`{card.get('direct_candidate')}`",
                f"- 摘要：{card.get('summary')}",
                f"- 检索词：{', '.join(card.get('retrieval_terms') or []) or '无'}",
                f"- 隐藏别名：{', '.join(card.get('hidden_retrieval_aliases') or []) or '无'}",
                f"- 查询扩展词：{', '.join(card.get('query_expansion_terms') or []) or '无'}",
                f"- 场景标签（不参与风格主分）：{', '.join(card.get('scene_tags') or []) or '无'}",
                f"- 人声/说唱/器乐/语言：{card.get('vocal_type') or '未设置'} / {card.get('rap_default')} / {card.get('instrumental_default')} / {card.get('default_language') or '未设置'}",
                f"- 来源：{source.get('title') or '缺失'}（{source.get('source_id') or 'missing'}；{source.get('license_status') or 'unknown'}）",
                f"- 来源链接：{source.get('url') or '无'}",
                f"- BGE 近邻观察条数：{len(card.get('embedding_observations') or [])}",
                f"- 影子策略观察条数：{len(card.get('policy_observations') or [])}",
                "",
                "### 审核问题",
                "",
            ]
        )
        lines.extend(f"- {question}" for question in card.get("review_questions", []))
        lines.extend(
            [
                "",
                "### 人工结论（待填写）",
                "",
                "- decision: `null`（可改为 `keep` / `revise` / `hold_for_source`）",
                "- notes: `null`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, default=Path("data/miaoxiang/style_templates.v2.jsonl"))
    parser.add_argument("--sources", type=Path, default=Path("data/miaoxiang/source_catalog.jsonl"))
    parser.add_argument("--embedding-result", type=Path, default=Path("data/output/evaluations/style-embedding-v1.1/result.json"))
    parser.add_argument("--policy-result", type=Path, default=Path("data/output/evaluations/style-policy-shadow-v1.2/result.json"))
    parser.add_argument("--ai-audit", type=Path, default=Path("data/miaoxiang/style_templates.v2.ai-audit.json"))
    parser.add_argument("--admission-audit", type=Path, default=Path("data/miaoxiang/style_templates.v2.admission-audit.json"))
    parser.add_argument("--out", type=Path, default=Path("data/output/evaluations/style-card-admission-review-v1-20260916"))
    args = parser.parse_args()

    cards = _read_jsonl(args.cards)
    sources = _read_jsonl(args.sources)
    embedding_result = _read_json(args.embedding_result) if args.embedding_result.exists() else None
    policy_result = _read_json(args.policy_result) if args.policy_result.exists() else None
    ai_audit = _read_json(args.ai_audit) if args.ai_audit.exists() else None
    admission_audit = _read_json(args.admission_audit) if args.admission_audit.exists() else None
    packet = prepare(cards, sources, embedding_result, policy_result, ai_audit, admission_audit)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "review_packet.md").write_text(render_markdown(packet), encoding="utf-8")
    print(
        json.dumps(
            {
                "packet_id": packet["packet_id"],
                "card_count": packet["card_count"],
                "group_count": packet["group_count"],
                "missing_source_card_ids": packet["missing_source_card_ids"],
                "ungrouped_card_ids": packet["ungrouped_card_ids"],
                "review_status_mutated": packet["review_status_mutated"],
                "production_index_changed": packet["production_index_changed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
