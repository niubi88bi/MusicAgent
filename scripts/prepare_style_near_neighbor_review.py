"""Prepare a human review packet for style-card near neighbors.

The packet is advisory: it exposes observed embedding neighbors and card
evidence, but it never changes review_status or makes an admission decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = [
    {
        "group_id": "dynamic_chinese_club",
        "label": "动感流行中文舞曲近邻",
        "card_ids": [
            "style-dynamic-vina-house-v2",
            "style-dynamic-chinese-bounce-v2",
            "style-dynamic-sad-slap-house-v2",
            "style-dynamic-sad-folk-pop-remix-v2",
        ],
        "query_ids": [
            "q01_dynamic_vina_house",
            "q02_dynamic_chinese_bounce",
            "q03_dynamic_sad_slap_house",
            "q04_dynamic_sad_folk_pop_remix",
        ],
        "review_questions": [
            "四张卡是否有足够明确的中文用户可感知差异？",
            "Vina House、Chinese Bounce、Sad Slap House 与项目组合是否会互相误召回？",
            "项目组合卡是否应保留为独立入口，还是降级为组合标签？",
        ],
    },
    {
        "group_id": "house_pair",
        "label": "Deep House / Tropical House 近邻",
        "card_ids": ["style-edm-deep-house-v2", "style-edm-tropical-house-v2"],
        "query_ids": ["q05_house_night_or_summer"],
        "review_questions": [
            "Rhodes/模拟低音与 Marimba/夏日氛围是否足以区分两张卡？",
            "仅输入‘浩室/四拍’时，是否应该返回两张候选而不是强行单选？",
        ],
    },
    {
        "group_id": "edm_energy_cluster",
        "label": "高速或旋律型 EDM 近邻",
        "card_ids": [
            "style-edm-future-bass-v2",
            "style-edm-trance-v2",
            "style-edm-drum-and-bass-v2",
            "style-edm-techno-v2",
        ],
        "query_ids": ["q06_edm_melodic_build_drop", "q07_edm_drum_and_bass"],
        "review_questions": [
            "Supersaw、琶音、breakbeat、工业底鼓等差异是否在卡片摘要中足够突出？",
            "Trance/Techno/DnB 的英文或 BPM 术语是否需要中文别名？",
            "Wikipedia 补充事实是否只支撑宽泛描述，没有被误写成后端控制能力？",
        ],
    },
    {
        "group_id": "chinese_voice_and_rap",
        "label": "国风与中文 Rap 近邻",
        "card_ids": [
            "style-guofeng-elegant-ballad-v2",
            "style-guofeng-epic-wuxia-v2",
            "style-guofeng-zen-ambient-v2",
            "style-guofeng-electronic-fusion-v2",
            "style-rap-zen-fusion-v2",
            "style-rap-cyber-zen-v2",
            "style-rap-healing-new-age-v2",
        ],
        "query_ids": ["q08_chinese_rap_and_guofeng"],
        "review_questions": [
            "用户说‘国风说唱’时，返回 Rap 卡、国风电子卡还是多候选更合理？",
            "古琴、木鱼、流水、琵琶等音色差异是否能避免近邻混淆？",
            "中文语种、说唱人声与国风意象是否被清楚分开？",
        ],
    },
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(cards: list[dict[str, Any]], embedding_result: dict[str, Any]) -> dict[str, Any]:
    by_id = {str(card["id"]): card for card in cards}
    observed = {row["query_id"]: row for row in embedding_result["embedding"]["results"]}
    groups: list[dict[str, Any]] = []
    for group in GROUPS:
        group_cards = []
        for card_id in group["card_ids"]:
            card = by_id[card_id]
            group_cards.append({
                "card_id": card_id,
                "summary": card["summary"],
                "retrieval_terms": card.get("retrieval_terms", []),
                "hidden_retrieval_aliases": card.get("hidden_retrieval_aliases", []),
                "evidence_depth": card.get("evidence_depth"),
                "source_id": card.get("source_id"),
                "review_status": card.get("review_status"),
            })
        query_observations = []
        for query_id in group["query_ids"]:
            row = observed[query_id]
            query_observations.append({
                "query_id": query_id,
                "query_text": row["query_text"],
                "top3": row["top3"],
                "primary_card_ids": row["primary_card_ids"],
                "primary_rank": row["primary_rank"],
            })
        groups.append({
            **group,
            "cards": group_cards,
            "observed_queries": query_observations,
            "human_decision": None,
            "human_notes": None,
        })
    return {
        "packet_id": "style-near-neighbor-review-v1-20260914",
        "packet_type": "human_review_pending",
        "source_evaluation": "data/output/evaluations/style-embedding-v1.1/result.json",
        "card_count": len(cards),
        "group_count": len(groups),
        "review_status_mutated": False,
        "groups": groups,
        "instructions": [
            "只判断卡片内容和近邻边界，不根据 top-1 命中率直接批准。",
            "human_decision 只能填写 keep、revise 或 hold_for_source。",
            "如果无法判断，保留 null，不把空白当作通过。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, default=Path("data/miaoxiang/style_templates.v2.jsonl"))
    parser.add_argument("--embedding-result", type=Path, default=Path("data/output/evaluations/style-embedding-v1.1/result.json"))
    parser.add_argument("--out", type=Path, default=Path("data/output/evaluations/style-near-neighbor-review-v1"))
    args = parser.parse_args()
    cards = _read_jsonl(args.cards)
    embedding_result = json.loads(args.embedding_result.read_text(encoding="utf-8"))
    packet = prepare(cards, embedding_result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "packet_id": packet["packet_id"],
        "card_count": packet["card_count"],
        "group_count": packet["group_count"],
        "review_status_mutated": packet["review_status_mutated"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
