"""Read-only retrieval from the isolated local-demo style collection."""

from __future__ import annotations

from pathlib import Path

from tools.rag.style_index import (
    ADMISSION_AUDIT_PATH,
    CARDS_PATH,
    COLLECTION_NAME,
    INDEX_PATH,
    MODEL_CACHE,
    MODEL_NAME,
    SOURCES_PATH,
    expand_composite_query,
    load_index_inputs,
    select_local_demo_cards,
)


def retrieve_local_demo_styles(
    query: str,
    top_k: int = 3,
    *,
    index_path: Path = INDEX_PATH,
    model_name: str = MODEL_NAME,
    cache_folder: Path = MODEL_CACHE,
) -> list[dict]:
    if not query.strip():
        raise ValueError("query_must_not_be_empty")
    if top_k < 1 or top_k > 5:
        raise ValueError("top_k_must_be_between_1_and_5")

    import chromadb
    from sentence_transformers import SentenceTransformer

    cards, sources, policy = load_index_inputs(
        CARDS_PATH, SOURCES_PATH, ADMISSION_AUDIT_PATH
    )
    direct, composite, blocked = select_local_demo_cards(cards, sources, policy)
    if blocked:
        raise RuntimeError(f"current_admission_state_is_blocked:{blocked}")
    direct_by_id = {card.id: card for card in direct}
    expanded_query, matched_template_ids = expand_composite_query(query, composite)

    client = chromadb.PersistentClient(path=str(index_path))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as error:
        raise RuntimeError("local_demo_style_collection_not_found") from error
    if collection.count() != len(direct):
        raise RuntimeError("local_demo_style_collection_count_mismatch")

    model = SentenceTransformer(model_name, cache_folder=str(cache_folder))
    embedding = model.encode(
        [expanded_query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    response = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=min(top_k, len(direct)),
        include=["distances", "metadatas"],
    )
    rows: list[dict] = []
    for card_id, distance, metadata in zip(
        response["ids"][0],
        response["distances"][0],
        response["metadatas"][0],
    ):
        card = direct_by_id.get(card_id)
        if card is None or metadata.get("source_scope") != "local_demo_allowed":
            raise RuntimeError("unauthorized_card_returned_from_local_demo_index")
        rows.append(
            {
                "card_id": card_id,
                "score": 1.0 - float(distance),
                "genre": card.genre,
                "subgenres": card.subgenres,
                "summary": card.summary,
                "source_id": card.source_id,
                "source_title": card.source_title,
                "source_url": card.source_url,
                "source_scope": metadata["source_scope"],
                "backend_validation": card.backend_validation.heartmula_local,
                "matched_composite_template_ids": matched_template_ids,
            }
        )
    return rows
