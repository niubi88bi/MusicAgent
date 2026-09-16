"""Compare BGE-small-zh embedding retrieval on the isolated style-card set.

This script is deliberately offline and draft-only. It builds a disposable
Chroma collection under the evaluation output directory and never changes the
production index or card review status.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from scripts.evaluate_style_fixed_queries import _read_jsonl, evaluate


def _document(card: dict[str, Any]) -> str:
    fields = (
        "id", "genre", "subgenres", "moods", "instruments", "melody",
        "dynamics", "structure_notes", "summary", "prompt_template",
        "retrieval_terms", "hidden_retrieval_aliases", "backend_tags",
        "scene_tags",
        "card_role", "retrieval_scope", "query_expansion_terms", "admission_note",
        "vocal_type", "rap_default", "instrumental_default", "default_language",
        "retrieval_policy_notes",
    )
    values: list[str] = []
    for field in fields:
        value = card.get(field)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif isinstance(value, dict):
            values.extend(str(item) for item in value.values())
        elif value is not None:
            values.append(str(value))
    return " | ".join(values)


def _evaluate_embedding(
    cards: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    manifest: dict[str, Any],
    model_name: str,
    collection_path: Path,
) -> dict[str, Any]:
    import chromadb

    model = SentenceTransformer(model_name)
    documents = [_document(card) for card in cards]
    embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
    queries = [query["query_text"] for query in manifest["queries"]]
    query_embeddings = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)

    collection_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(collection_path))
    collection = client.get_or_create_collection(
        name="style_draft_v11",
        metadata={"hnsw:space": "cosine", "evaluation_only": True},
    )
    # upsert keeps rerunning the same isolated evaluation idempotent.
    collection.upsert(
        ids=[str(card["id"]) for card in cards],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source_id": str(card["source_id"]), "review_status": str(card["review_status"])}
            for card in cards
        ],
    )
    response = collection.query(
        query_embeddings=query_embeddings.tolist(),
        n_results=min(3, len(cards)),
        include=["distances", "metadatas"],
    )
    source_ids = {str(row["source_id"]) for row in sources}
    card_by_id = {str(card["id"]): card for card in cards}
    rows: list[dict[str, Any]] = []
    for index, query in enumerate(manifest["queries"]):
        ranked_ids = response["ids"][index]
        distances = response["distances"][index]
        ranked = [
            {
                "card_id": card_id,
                "cosine_distance": distance,
                "source_id": card_by_id[card_id]["source_id"],
                "review_status": card_by_id[card_id]["review_status"],
            }
            for card_id, distance in zip(ranked_ids, distances)
        ]
        primary = set(query["primary_card_ids"])
        expected = set(query["expected_card_ids"])
        primary_rank = next((i + 1 for i, row in enumerate(ranked) if row["card_id"] in primary), None)
        top1_hit = bool(ranked and ranked[0]["card_id"] in primary)
        top3_hit = any(row["card_id"] in expected for row in ranked[:3])
        traceable = all(row["source_id"] in source_ids for row in ranked)
        rows.append({
            "query_id": query["query_id"],
            "query_text": query["query_text"],
            "primary_card_ids": sorted(primary),
            "expected_card_ids": sorted(expected),
            "top3": ranked,
            "primary_rank": primary_rank,
            "top1_primary_hit": top1_hit,
            "top3_expected_hit": top3_hit,
            "top3_source_traceable": traceable,
        })

    return {
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "collection_path": str(collection_path),
        "collection_name": collection.name,
        "query_count": len(rows),
        "top1_primary_accuracy": sum(row["top1_primary_hit"] for row in rows) / len(rows),
        "top3_expected_recall": sum(row["top3_expected_hit"] for row in rows) / len(rows),
        "mean_primary_reciprocal_rank": sum((1 / row["primary_rank"]) if row["primary_rank"] else 0 for row in rows) / len(rows),
        "top3_source_traceability": all(row["top3_source_traceable"] for row in rows),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/miaoxiang/style_fixed_queries.v1.1.json"))
    parser.add_argument("--cards", type=Path, default=Path("data/miaoxiang/style_templates.v2.jsonl"))
    parser.add_argument("--sources", type=Path, default=Path("data/miaoxiang/source_catalog.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/output/evaluations/style-embedding-v1.1"))
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cards = _read_jsonl(args.cards)
    sources = _read_jsonl(args.sources)
    lexical = evaluate(cards, sources, manifest)
    embedding = _evaluate_embedding(cards, sources, manifest, args.model, args.out / "chroma")
    result = {
        "evaluation_id": f"{manifest['query_set_id']}-embedding-{date.today().isoformat()}",
        "manifest": str(args.manifest),
        "cards": str(args.cards),
        "sources": str(args.sources),
        "evaluated_at": date.today().isoformat(),
        "retrieval_mode": "bge_embedding_with_isolated_chroma",
        "card_count": len(cards),
        "source_count": len(sources),
        "draft_card_count": sum(card.get("review_status") == "draft" for card in cards),
        "runtime": {"python": platform.python_version(), "chromadb": version("chromadb"), "sentence_transformers": version("sentence-transformers")},
        "lexical_baseline": {
            "result_path": "data/output/evaluations/style-fixed-query-v1.1/result.json",
            "top1_primary_accuracy": lexical["top1_primary_accuracy"],
            "top3_expected_recall": lexical["top3_expected_recall"],
            "mean_primary_reciprocal_rank": lexical["mean_primary_reciprocal_rank"],
        },
        "embedding": embedding,
        "production_index_changed": False,
        "production_cards_changed": False,
        "interpretation": "isolated draft-card comparison only; no claim about generation quality, backend control, or production RAG value",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "evaluation_id": result["evaluation_id"],
        "model_name": embedding["model_name"],
        "embedding_dimension": embedding["embedding_dimension"],
        "lexical_top1": lexical["top1_primary_accuracy"],
        "embedding_top1": embedding["top1_primary_accuracy"],
        "lexical_top3": lexical["top3_expected_recall"],
        "embedding_top3": embedding["top3_expected_recall"],
        "embedding_traceable": embedding["top3_source_traceability"],
        "production_index_changed": False,
    }, ensure_ascii=False, indent=2))
    for row in embedding["results"]:
        top = row["top3"][0]
        print(f"{row['query_id']}: top1={top['card_id']} distance={top['cosine_distance']:.4f} primary_hit={row['top1_primary_hit']} primary_rank={row['primary_rank']}")


if __name__ == "__main__":
    main()
