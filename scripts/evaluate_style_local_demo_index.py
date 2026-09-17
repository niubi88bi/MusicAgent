"""Evaluate the isolated local-demo style collection with registered queries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.rag.style_index import (
    COLLECTION_NAME,
    INDEX_PATH,
    MODEL_CACHE,
    MODEL_NAME,
    expand_composite_query,
    load_index_inputs,
    select_local_demo_cards,
)


DEFAULT_MANIFESTS = [
    Path("data/miaoxiang/style_fixed_queries.v1.1.json"),
    Path("data/miaoxiang/style_fixed_queries.v1.2.json"),
    Path("data/miaoxiang/style_fixed_queries.v1.3.json"),
]


def _labels(query: dict[str, Any], direct_ids: set[str]) -> tuple[set[str], set[str], set[str]]:
    templates = set(query.get("expected_template_ids", []))
    primary = set(query.get("primary_card_ids", [])) & direct_ids
    expected = set(
        query.get("expected_direct_card_ids", query.get("expected_card_ids", []))
    ) & direct_ids
    return templates, primary, expected


def evaluate_rankings(
    manifest: dict[str, Any],
    rankings: list[list[dict[str, Any]]],
    matched_templates: list[list[str]],
    direct_ids: set[str],
    valid_source_ids: set[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query, ranked, matched in zip(manifest["queries"], rankings, matched_templates):
        expected_templates, primary, expected = _labels(query, direct_ids)
        top_ids = [row["card_id"] for row in ranked]
        rows.append(
            {
                "query_id": query["query_id"],
                "query_mode": query.get("query_mode", "direct_style"),
                "query_text": query["query_text"],
                "matched_composite_template_ids": matched,
                "template_expected_hit": bool(expected_templates) and bool(
                    expected_templates & set(matched)
                ),
                "direct_primary_ids": sorted(primary),
                "direct_expected_ids": sorted(expected),
                "top3": ranked,
                "top1_primary_hit": bool(primary) and bool(top_ids) and top_ids[0] in primary,
                "top3_expected_hit": bool(expected) and bool(expected & set(top_ids[:3])),
                "source_traceable": all(
                    row["source_id"] in valid_source_ids for row in ranked
                ),
                "scope_valid": all(row["source_scope"] == "local_demo_allowed" for row in ranked),
                "composite_returned": any(card_id not in direct_ids for card_id in top_ids),
            }
        )
    primary_rows = [row for row in rows if row["direct_primary_ids"]]
    expected_rows = [row for row in rows if row["direct_expected_ids"]]
    template_rows = [
        row for row in rows if row["query_mode"] == "composite_expansion"
    ]
    return {
        "manifest_id": manifest["query_set_id"],
        "query_count": len(rows),
        "direct_primary_query_count": len(primary_rows),
        "direct_expected_query_count": len(expected_rows),
        "direct_top1_primary_accuracy": (
            sum(row["top1_primary_hit"] for row in primary_rows) / len(primary_rows)
            if primary_rows
            else None
        ),
        "direct_top3_expected_recall": (
            sum(row["top3_expected_hit"] for row in expected_rows) / len(expected_rows)
            if expected_rows
            else None
        ),
        "composite_template_accuracy": (
            sum(row["template_expected_hit"] for row in template_rows) / len(template_rows)
            if template_rows
            else None
        ),
        "source_traceability": all(row["source_traceable"] for row in rows),
        "scope_valid": all(row["scope_valid"] for row in rows),
        "composite_returned_count": sum(row["composite_returned"] for row in rows),
        "results": rows,
    }


def evaluate_index(
    manifest_paths: list[Path],
    index_path: Path = INDEX_PATH,
    model_name: str = MODEL_NAME,
    cache_folder: Path = MODEL_CACHE,
) -> dict[str, Any]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    cards, sources, policy = load_index_inputs()
    direct, composite, blocked = select_local_demo_cards(cards, sources, policy)
    if blocked:
        raise RuntimeError(f"current_admission_state_is_blocked:{blocked}")
    direct_ids = {card.id for card in direct}
    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.get_collection(COLLECTION_NAME)
    if collection.count() != len(direct):
        raise RuntimeError("local_demo_style_collection_count_mismatch")

    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    all_queries: list[str] = []
    all_template_matches: list[list[str]] = []
    manifest_slices: list[tuple[int, int]] = []
    for manifest in manifests:
        start = len(all_queries)
        for query in manifest["queries"]:
            expanded, matched = expand_composite_query(query["query_text"], composite)
            all_queries.append(expanded)
            all_template_matches.append(matched)
        manifest_slices.append((start, len(all_queries)))

    model = SentenceTransformer(model_name, cache_folder=str(cache_folder))
    embeddings = model.encode(
        all_queries,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    response = collection.query(
        query_embeddings=embeddings.tolist(),
        n_results=3,
        include=["distances", "metadatas"],
    )
    rankings: list[list[dict[str, Any]]] = []
    for ids, distances, metadatas in zip(
        response["ids"], response["distances"], response["metadatas"]
    ):
        rankings.append(
            [
                {
                    "card_id": card_id,
                    "score": 1.0 - float(distance),
                    "source_id": metadata.get("primary_source_id"),
                    "source_scope": metadata.get("source_scope"),
                }
                for card_id, distance, metadata in zip(ids, distances, metadatas)
            ]
        )

    manifest_results = []
    for manifest, (start, end) in zip(manifests, manifest_slices):
        manifest_results.append(
            evaluate_rankings(
                manifest,
                rankings[start:end],
                all_template_matches[start:end],
                direct_ids,
                set(sources),
            )
        )
    return {
        "evaluation_id": "style-local-demo-index-v1",
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "collection_name": COLLECTION_NAME,
        "collection_count": collection.count(),
        "embedding_model": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "direct_card_count": len(direct),
        "composite_template_count": len(composite),
        "public_product_eligible_count": 0,
        "agent_connected": False,
        "manifests": manifest_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", dest="manifests")
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cache-folder", type=Path, default=MODEL_CACHE)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/output/evaluations/style-local-demo-index-v1"),
    )
    args = parser.parse_args()
    result = evaluate_index(
        args.manifests or DEFAULT_MANIFESTS,
        args.index_path,
        args.model,
        args.cache_folder,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "query_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "evaluation_id": result["evaluation_id"],
                "collection_count": result["collection_count"],
                "embedding_dimension": result["embedding_dimension"],
                "public_product_eligible_count": result["public_product_eligible_count"],
                "agent_connected": result["agent_connected"],
                "manifests": [
                    {
                        key: row[key]
                        for key in (
                            "manifest_id",
                            "query_count",
                            "direct_top1_primary_accuracy",
                            "direct_top3_expected_recall",
                            "composite_template_accuracy",
                            "source_traceability",
                            "scope_valid",
                            "composite_returned_count",
                        )
                    }
                    for row in result["manifests"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
