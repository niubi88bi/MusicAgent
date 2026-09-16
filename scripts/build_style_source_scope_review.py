"""Build a human review packet for style-card source-use scopes.

The packet summarizes the existing local source catalog and card metadata.  It
does not perform a legal review, approve a source, or mutate card/audit state.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.rag.schemas import SourceRecord, StyleCard


DEFAULT_CARDS_PATH = Path("data/miaoxiang/style_templates.v2.jsonl")
DEFAULT_SOURCES_PATH = Path("data/miaoxiang/source_catalog.jsonl")
DEFAULT_ADMISSION_AUDIT_PATH = Path(
    "data/miaoxiang/style_templates.v2.admission-audit.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/output/evaluations/style-source-scope-review-v1-20260916"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_review_packet(
    cards_path: Path = DEFAULT_CARDS_PATH,
    sources_path: Path = DEFAULT_SOURCES_PATH,
    admission_audit_path: Path = DEFAULT_ADMISSION_AUDIT_PATH,
) -> dict[str, Any]:
    cards = [StyleCard.model_validate(row) for row in _read_jsonl(cards_path)]
    sources = [SourceRecord.model_validate(row) for row in _read_jsonl(sources_path)]
    audit = json.loads(admission_audit_path.read_text(encoding="utf-8"))
    content_decisions = {
        row["card_id"]: row["decision"]
        for row in audit.get("human_card_review", {}).get("cards", [])
    }

    primary_cards: dict[str, list[str]] = defaultdict(list)
    supplemental_cards: dict[str, list[str]] = defaultdict(list)
    supplemental_details: dict[str, list[dict[str, str]]] = defaultdict(list)
    for card in cards:
        primary_cards[card.source_id].append(card.id)
        for supplemental in card.supplemental_sources:
            supplemental_cards[supplemental.source_id].append(card.id)
            supplemental_details[supplemental.source_id].append(
                {
                    "card_id": card.id,
                    "attribution": supplemental.attribution,
                    "changes": supplemental.changes,
                    "license_status": supplemental.license_status,
                }
            )

    source_reviews: list[dict[str, Any]] = []
    for source in sources:
        is_share_alike = source.license_status == "CC-BY-SA-4.0"
        source_reviews.append(
            {
                "source_id": source.source_id,
                "source_title": source.title,
                "source_type": source.source_type,
                "publisher": source.publisher,
                "url": source.url,
                "verification_status": source.verification_status,
                "license_status": source.license_status,
                "recorded_allowed_project_use": source.allowed_project_use,
                "raw_content_saved": source.raw_content_saved,
                "primary_card_ids": sorted(primary_cards[source.source_id]),
                "supplemental_card_ids": sorted(supplemental_cards[source.source_id]),
                "design_reference_only": not (
                    primary_cards[source.source_id] or supplemental_cards[source.source_id]
                ),
                "stored_content_summary": [
                    "来源目录元数据",
                    "可核对的事实性字段",
                    "项目原创摘要或字段设计说明",
                ],
                "excluded_content_guardrails": [
                    "网页全文",
                    "完整示例歌词",
                    "图片",
                    "音频",
                ],
                "attribution_required": is_share_alike,
                "share_alike_required": is_share_alike,
                "supplemental_attribution_records": supplemental_details[source.source_id],
                "non_binding_recommended_scope": "evaluation_only",
                "decision": None,
                "review_notes": "",
            }
        )

    card_reviews: list[dict[str, Any]] = []
    for card in cards:
        card_reviews.append(
            {
                "card_id": card.id,
                "card_role": card.card_role,
                "retrieval_scope": card.retrieval_scope,
                "content_review_decision": content_decisions.get(card.id),
                "primary_source_id": card.source_id,
                "supplemental_source_ids": [
                    source.source_id for source in card.supplemental_sources
                ],
                "stored_card_content": [
                    "事实性字段",
                    "项目原创 summary",
                    "项目原创 prompt_template",
                    "项目派生检索词与后端候选标签",
                ],
                "backend_validation": card.backend_validation.heartmula_local,
                "non_binding_recommended_scope": "evaluation_only",
                "decision": None,
                "review_notes": "",
            }
        )

    return {
        "schema_version": "1.0",
        "review_id": "style-source-scope-review-v1-20260916",
        "review_mode": "human_decision_required",
        "scope_options": [
            "evaluation_only",
            "local_demo_allowed",
            "public_product_allowed",
            "blocked",
        ],
        "legal_boundary": (
            "本审核包只汇总项目已登记证据与风险提示，不构成法律意见；"
            "推荐范围不产生授权效果，只有人工 decision 才能进入后续准入审计。"
        ),
        "summary": {
            "source_count": len(source_reviews),
            "card_count": len(card_reviews),
            "source_decisions_pending": sum(
                row["decision"] is None for row in source_reviews
            ),
            "card_decisions_pending": sum(row["decision"] is None for row in card_reviews),
            "local_demo_eligible_count": 0,
            "public_product_eligible_count": 0,
            "review_status_changed": False,
            "index_changed": False,
        },
        "source_reviews": source_reviews,
        "card_reviews": card_reviews,
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# 风格卡来源使用范围人工审核包",
        "",
        f"> {packet['legal_boundary']}",
        "",
        "## 来源审核",
        "",
        "| 来源 | 许可记录 | 关联卡片 | 建议范围 | 人工决定 |",
        "|---|---|---:|---|---|",
    ]
    for row in packet["source_reviews"]:
        linked_count = len(row["primary_card_ids"]) + len(row["supplemental_card_ids"])
        lines.append(
            f"| `{row['source_id']}` | {row['license_status']} | {linked_count} | "
            f"{row['non_binding_recommended_scope']} | 待填写 |"
        )
    lines.extend(
        [
            "",
            "## 卡片审核",
            "",
            "| 卡片 | 内容结论 | 主来源 | 角色 | 建议范围 | 人工决定 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in packet["card_reviews"]:
        lines.append(
            f"| `{row['card_id']}` | {row['content_review_decision']} | "
            f"`{row['primary_source_id']}` | {row['card_role']} | "
            f"{row['non_binding_recommended_scope']} | 待填写 |"
        )
    lines.extend(
        [
            "",
            "## 填写规则",
            "",
            "- 只能填写 `evaluation_only`、`local_demo_allowed`、`public_product_allowed` 或 `blocked`。",
            "- `keep`、固定查询通过和 schema 通过都不等于来源授权。",
            "- Wikipedia 补充卡必须继续保留固定版本、署名、修改说明和 CC BY-SA 4.0 提示。",
            "- 本轮不修改卡片 `review_status`，也不创建任何索引。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS_PATH)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_PATH)
    parser.add_argument("--admission-audit", type=Path, default=DEFAULT_ADMISSION_AUDIT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    packet = build_review_packet(args.cards, args.sources, args.admission_audit)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "review_packet.md").write_text(
        render_markdown(packet),
        encoding="utf-8",
    )
    print(json.dumps(packet["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
