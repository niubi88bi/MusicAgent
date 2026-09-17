"""Build the isolated local-demo Chroma style index."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from tools.rag.style_index import (
    ADMISSION_AUDIT_PATH,
    CARDS_PATH,
    COLLECTION_NAME,
    INDEX_PATH,
    MODEL_CACHE,
    MODEL_NAME,
    SOURCES_PATH,
    build_embedding_document,
    build_metadata,
    load_index_inputs,
    select_local_demo_cards,
)


def prepare_build_report(
    cards_path: Path = CARDS_PATH,
    sources_path: Path = SOURCES_PATH,
    admission_audit_path: Path = ADMISSION_AUDIT_PATH,
) -> tuple[dict, list, list]:
    cards, sources, policy = load_index_inputs(cards_path, sources_path, admission_audit_path)
    direct, composite, blocked = select_local_demo_cards(cards, sources, policy)
    report = {
        "scope": "local_demo",
        "collection_name": COLLECTION_NAME,
        "card_count": len(cards),
        "source_count": len(sources),
        "direct_candidate_count": len(direct),
        "composite_template_count": len(composite),
        "blocked_non_composite_count": len(blocked),
        "direct_candidate_ids": [card.id for card in direct],
        "composite_template_ids": [card.id for card in composite],
        "blocked_cards": blocked,
        "all_direct_cards_approved": all(card.review_status == "approved" for card in direct),
        "public_product_eligible_count": 0,
        "index_changed": False,
    }
    return report, direct, composite


def build_index(
    index_path: Path,
    model_name: str,
    cache_folder: Path,
    cards_path: Path = CARDS_PATH,
    sources_path: Path = SOURCES_PATH,
    admission_audit_path: Path = ADMISSION_AUDIT_PATH,
) -> dict:
    import chromadb
    from sentence_transformers import SentenceTransformer

    report, direct, composite = prepare_build_report(
        cards_path, sources_path, admission_audit_path
    )
    if report["blocked_non_composite_count"]:
        raise RuntimeError(f"Local-demo admission blocked: {report['blocked_cards']}")
    if len(direct) != 16 or len(composite) != 1:
        raise RuntimeError(
            f"Unexpected admitted set: direct={len(direct)} composite={len(composite)}"
        )
    index_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_path))
    if COLLECTION_NAME in {row.name for row in client.list_collections()}:
        raise RuntimeError(
            f"Collection {COLLECTION_NAME} already exists; refusing to overwrite it"
        )
    model = SentenceTransformer(model_name, cache_folder=str(cache_folder))
    documents = [build_embedding_document(card) for card in direct]
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "scope": "local_demo"},
    )
    collection.add(
        ids=[card.id for card in direct],
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=[build_metadata(card) for card in direct],
    )
    report.update(
        {
            "embedding_model": model_name,
            "embedding_dimension": int(embeddings.shape[1]),
            "index_path": str(index_path),
            "collection_count": collection.count(),
            "content_hashes": {card.id: card.content_sha256 for card in direct},
            "created_at": datetime.now().astimezone().isoformat(),
            "runtime": {
                "python": platform.python_version(),
                "chromadb": version("chromadb"),
                "sentence_transformers": version("sentence-transformers"),
            },
            "index_changed": True,
            "agent_connected": False,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["local-demo"], default="local-demo")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cards", type=Path, default=CARDS_PATH)
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument("--admission-audit", type=Path, default=ADMISSION_AUDIT_PATH)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cache-folder", type=Path, default=MODEL_CACHE)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/output/evaluations/style-local-demo-index-v1"),
    )
    args = parser.parse_args()

    if args.execute:
        report = build_index(
            args.index_path,
            args.model,
            args.cache_folder,
            args.cards,
            args.sources,
            args.admission_audit,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "index_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        report, _, _ = prepare_build_report(
            args.cards, args.sources, args.admission_audit
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
