"""Evaluate role-aware style retrieval policies without production side effects."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from scripts.evaluate_style_fixed_queries import _read_jsonl, _score


def _policy_score(query: dict[str, Any], card: dict[str, Any]) -> tuple[int, list[str]]:
    score, matched = _score(query, card)
    policy = query.get("policy", {})
    if policy.get("require_rap"):
        if card.get("rap_default"):
            score += 4
            matched.append("policy:rap_default")
        if card.get("instrumental_default"):
            score -= 4
    if policy.get("require_instrumental"):
        if card.get("instrumental_default"):
            score += 4
            matched.append("policy:instrumental_default")
        if card.get("rap_default"):
            score -= 4
    return score, matched


def _rank(query: dict[str, Any], cards: list[dict[str, Any]], *, direct_only: bool) -> list[dict[str, Any]]:
    candidates = [
        card for card in cards
        if not direct_only or card.get("retrieval_scope", "direct_candidate") == "direct_candidate"
    ]
    family = query.get("policy", {}).get("family")
    if family == "guofeng":
        candidates = [
            card for card in candidates
            if card.get("genre") == "民族国韵" or any("国风" in str(value) for value in card.get("subgenres", []))
        ]
    ranked = []
    for card in candidates:
        score, matched = _policy_score(query, card)
        ranked.append({"card_id": card["id"], "score": score, "matched": matched, "card_role": card.get("card_role", "style_card")})
    ranked.sort(key=lambda row: (-row["score"], row["card_id"]))
    return ranked


def evaluate(cards: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for query in manifest["queries"]:
        mode = query["query_mode"]
        template_ranked = _rank(query, cards, direct_only=False)
        direct_ranked = _rank(query, cards, direct_only=True)
        template_ids = set(query.get("expected_template_ids", []))
        expected = set(query.get("expected_direct_card_ids", []))
        primary = set(query.get("primary_card_ids", []))
        template_top = template_ranked[0]["card_id"] if template_ranked else None
        primary_rank = next((i + 1 for i, row in enumerate(direct_ranked) if row["card_id"] in primary), None) if primary else None
        expected_count = sum(row["card_id"] in expected for row in direct_ranked[:3])
        required_count = int(query.get("policy", {}).get("min_expected_top3", 1))
        rows.append({
            "query_id": query["query_id"],
            "query_mode": mode,
            "query_text": query["query_text"],
            "policy": query.get("policy", {}),
            "template_top": template_ranked[0] if template_ranked else None,
            "template_expected_hit": bool(template_ids and template_top in template_ids),
            "direct_top3": direct_ranked[:3],
            "direct_primary_rank": primary_rank,
            "direct_primary_hit": bool(primary and direct_ranked and direct_ranked[0]["card_id"] in primary),
            "direct_expected_top3_count": expected_count,
            "direct_expected_top3_hit": expected_count >= required_count,
            "direct_excludes_composite": all(row["card_role"] != "composite_query_template" for row in direct_ranked[:3]),
        })
    direct_queries = [row for row in rows if row["query_mode"] != "composite_expansion"]
    primary_queries = [row for row in direct_queries if row["direct_primary_rank"] is not None]
    return {
        "evaluation_id": f"{manifest['query_set_id']}-run-{date.today().isoformat()}",
        "manifest_id": manifest["query_set_id"],
        "retrieval_mode": manifest["retrieval_mode"],
        "card_count": len(cards),
        "query_count": len(rows),
        "composite_template_top1_accuracy": sum(row["template_expected_hit"] for row in rows if row["query_mode"] == "composite_expansion") / max(1, sum(row["query_mode"] == "composite_expansion" for row in rows)),
        "direct_primary_top1_accuracy": sum(row["direct_primary_hit"] for row in primary_queries) / max(1, len(primary_queries)),
        "direct_expected_top3_recall": sum(row["direct_expected_top3_hit"] for row in direct_queries) / max(1, len(direct_queries)),
        "direct_top3_excludes_composite": all(row["direct_excludes_composite"] for row in rows),
        "production_index_changed": False,
        "review_status_changed": False,
        "results": rows,
        "interpretation": "影子规则只验证组合模板和人声属性的候选排序，不代表生产路由或音频控制已验证。",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/miaoxiang/style_fixed_queries.v1.2.json"))
    parser.add_argument("--cards", type=Path, default=Path("data/miaoxiang/style_templates.v2.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/output/evaluations/style-policy-shadow-v1.2"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cards = _read_jsonl(args.cards)
    result = evaluate(cards, manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "evaluation_id", "card_count", "query_count", "composite_template_top1_accuracy",
        "direct_primary_top1_accuracy", "direct_expected_top3_recall",
        "direct_top3_excludes_composite", "production_index_changed", "review_status_changed",
    )}, ensure_ascii=False, indent=2))
    for row in result["results"]:
        top = row["direct_top3"][0] if row["direct_top3"] else None
        print(f"{row['query_id']}: mode={row['query_mode']} top1={top['card_id'] if top else None} primary_hit={row['direct_primary_hit']} expected_count={row['direct_expected_top3_count']} expected_top3={row['direct_expected_top3_hit']}")


if __name__ == "__main__":
    main()
