"""Evaluate the pre-registered style queries without installing a vector stack.

This is a deterministic lexical baseline for draft-card admission only. It is
not an embedding benchmark and never changes review_status or production data.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalise(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _card_blob(card: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Return searchable text and deterministic weights for term classes.

    Core style terms and subgenres are strongest. Hidden aliases are useful for
    recall but must not outrank a named style; query-expansion terms are weaker,
    and scene tags are context-only (zero style score).
    """
    fields = (
        "id", "genre", "subgenres", "moods", "instruments", "melody",
        "dynamics", "structure_notes", "summary", "prompt_template",
        "retrieval_terms", "hidden_retrieval_aliases", "query_expansion_terms",
        "scene_tags", "backend_tags",
    )
    values: list[str] = []
    term_weights: dict[str, int] = {}
    for field in fields:
        value = card.get(field)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
            if field in {"retrieval_terms", "subgenres"}:
                for item in value:
                    term_weights[_normalise(str(item))] = max(
                        term_weights.get(_normalise(str(item)), 0), 4
                    )
            elif field == "hidden_retrieval_aliases":
                for item in value:
                    token = _normalise(str(item))
                    term_weights[token] = max(term_weights.get(token, 0), 2)
            elif field == "query_expansion_terms":
                for item in value:
                    token = _normalise(str(item))
                    term_weights[token] = max(term_weights.get(token, 0), 1)
            elif field == "scene_tags":
                for item in value:
                    token = _normalise(str(item))
                    term_weights.setdefault(token, 0)
        elif isinstance(value, dict):
            values.extend(str(item) for item in value.values())
        elif value is not None:
            values.append(str(value))
    return _normalise(" ".join(values)), term_weights


def _score(query: dict[str, Any], card: dict[str, Any]) -> tuple[int, list[str]]:
    blob, term_weights = _card_blob(card)
    score = 0
    matched: list[str] = []
    for keyword in query["keywords"]:
        token = _normalise(keyword)
        if not token or token not in blob:
            continue
        weight = term_weights.get(token, 1)
        if weight > 0:
            matched.append(keyword)
            score += weight
    return score, matched


def evaluate(cards: list[dict[str, Any]], sources: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    source_ids = {str(row["source_id"]) for row in sources}
    card_by_id = {str(card["id"]): card for card in cards}
    rows: list[dict[str, Any]] = []
    for query in manifest["queries"]:
        ranked = []
        for card in cards:
            score, matched = _score(query, card)
            ranked.append({"card_id": card["id"], "score": score, "matched_keywords": matched})
        ranked.sort(key=lambda row: (-row["score"], row["card_id"]))
        top3 = ranked[:3]
        primary = set(query["primary_card_ids"])
        expected = set(query["expected_card_ids"])
        top1_hit = bool(ranked and ranked[0]["card_id"] in primary)
        top3_hit = any(row["card_id"] in expected for row in top3)
        primary_rank = next((i + 1 for i, row in enumerate(ranked) if row["card_id"] in primary), None)
        top1_score = ranked[0]["score"] if ranked else 0
        top2_score = ranked[1]["score"] if len(ranked) > 1 else 0
        rows.append({
            "query_id": query["query_id"],
            "query_text": query["query_text"],
            "primary_card_ids": sorted(primary),
            "expected_card_ids": sorted(expected),
            "top1": ranked[0] if ranked else None,
            "top3": top3,
            "primary_rank": primary_rank,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "top1_margin": top1_score - top2_score,
            "low_signal": top1_score < 4,
            "top1_primary_hit": top1_hit,
            "top3_expected_hit": top3_hit,
        })

    traceable = all(card_by_id[row["top1"]["card_id"]].get("source_id") in source_ids for row in rows if row["top1"])
    return {
        "evaluation_id": f"{manifest['query_set_id']}-run-{date.today().isoformat()}",
        "manifest_id": manifest["query_set_id"],
        "evaluated_at": date.today().isoformat(),
        "retrieval_mode": manifest["retrieval_mode"],
        "card_count": len(cards),
        "source_count": len(sources),
        "draft_card_count": sum(card.get("review_status") == "draft" for card in cards),
        "approved_card_count": sum(card.get("review_status") == "approved" for card in cards),
        "query_count": len(rows),
        "top1_primary_accuracy": sum(row["top1_primary_hit"] for row in rows) / len(rows),
        "top3_expected_recall": sum(row["top3_expected_hit"] for row in rows) / len(rows),
        "mean_primary_reciprocal_rank": sum((1 / row["primary_rank"]) if row["primary_rank"] else 0 for row in rows) / len(rows),
        "queries_with_top1_score_ge_4": sum(row["top1_score"] >= 4 for row in rows),
        "low_signal_query_count": sum(row["low_signal"] for row in rows),
        "minimum_top1_margin": min(row["top1_margin"] for row in rows),
        "top1_source_traceability": traceable,
        "production_index_changed": False,
        "results": rows,
        "interpretation": "词面基线仅用于检查固定查询与草稿卡的登记语义；不能代表 embedding 检索、生成质量或生产效果。",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/miaoxiang/style_fixed_queries.v1.json"))
    parser.add_argument("--cards", type=Path, default=Path("data/miaoxiang/style_templates.v2.jsonl"))
    parser.add_argument("--sources", type=Path, default=Path("data/miaoxiang/source_catalog.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/output/evaluations/style-fixed-query-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cards = _read_jsonl(args.cards)
    sources = _read_jsonl(args.sources)
    result = evaluate(cards, sources, manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "evaluation_id", "card_count", "source_count", "draft_card_count",
        "approved_card_count",
        "query_count", "top1_primary_accuracy", "top3_expected_recall",
        "mean_primary_reciprocal_rank", "queries_with_top1_score_ge_4",
        "low_signal_query_count", "minimum_top1_margin", "top1_source_traceability",
        "production_index_changed",
    )}, ensure_ascii=False, indent=2))
    for row in result["results"]:
        print(f"{row['query_id']}: top1={row['top1']['card_id']} score={row['top1_score']} margin={row['top1_margin']} low_signal={row['low_signal']} primary_hit={row['top1_primary_hit']} primary_rank={row['primary_rank']}")


if __name__ == "__main__":
    main()
