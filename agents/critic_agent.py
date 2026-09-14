"""匿名比较双作词候选的主观评审 Agent。"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

from agents.schemas import (
    CandidateSubjectiveReview,
    JsonLLM,
    LyricDraft,
    LyricJudgeResponse,
    SongRequirement,
    pydantic_validation_feedback,
)


SUBJECTIVE_WEIGHTS = {
    "theme_fit": 25,
    "naturalness": 20,
    "emotional_control": 20,
    "narrative_imagery_coherence": 20,
    "hook_memorability": 15,
}
SUBJECTIVE_NEAR_TIE_POINTS = 3.0
MAX_CANDIDATE_REGENERATIONS = 1


def weighted_subjective_score(review: CandidateSubjectiveReview) -> float:
    """由程序计算 0-100 主观总分，避免依赖模型自行做加权算术。"""

    total = sum(
        getattr(review, field_name).score / 5 * weight
        for field_name, weight in SUBJECTIVE_WEIGHTS.items()
    )
    return round(total, 1)


def has_major_theme_issue(review: CandidateSubjectiveReview) -> bool:
    """低主题分或明确的重大主题偏移，均视为不可直接入选。"""

    if review.theme_fit.score < 3.0:
        return True
    return any(
        issue.code == "theme_drift" and issue.severity == "major"
        for issue in review.issues
    )


def format_subjective_revision_feedback(review: Mapping[str, object]) -> str:
    """把匿名评审问题压缩为可交给对应 persona 的定向改写反馈。"""

    issues = review.get("issues")
    if not isinstance(issues, list) or not issues:
        return "主观评审认为主题方向存在明显问题，请严格对照歌曲需求重写核心叙事与 Hook。"
    lines = ["主观评审认为当前版本存在明显问题，请按证据定向修改："]
    for issue in issues[:4]:
        if not isinstance(issue, dict):
            continue
        lines.append(
            f"- [{issue.get('code', 'other')}] 证据：{issue.get('evidence', '未提供')}；"
            f"建议：{issue.get('suggestion', '请修正')}"
        )
    return "\n".join(lines)


def select_subjective_candidate(
    candidates: Sequence[Mapping[str, object]],
    review: Mapping[str, object],
    candidate_order: Mapping[str, int],
) -> tuple[Mapping[str, object], str]:
    """按已确认的重大问题、主观分差和规则兜底顺序选择候选。"""

    if len(candidates) != 2:
        raise ValueError("主观选优必须收到两个候选")
    reviews = review.get("candidate_reviews")
    if not isinstance(reviews, dict):
        raise TypeError("主观评审候选分数缺失")

    ordered = sorted(
        candidates,
        key=lambda item: candidate_order[str(item["candidate_id"])],
    )

    def rule_rank(candidate: Mapping[str, object]) -> tuple[bool, float, float, float, int]:
        report = candidate.get("quality_report")
        scores = candidate.get("scores")
        if not isinstance(report, dict) or not isinstance(scores, dict):
            raise TypeError("作词候选缺少规则报告")
        candidate_id = str(candidate["candidate_id"])
        return (
            bool(report.get("passed")),
            float(scores["total"]),
            float(scores["rhyme"]),
            float(scores["structure"]),
            -candidate_order[candidate_id],
        )

    major: dict[str, bool] = {}
    for candidate in ordered:
        candidate_id = str(candidate["candidate_id"])
        candidate_review = reviews.get(candidate_id)
        if not isinstance(candidate_review, dict):
            raise TypeError(f"主观评审缺少候选 {candidate_id} 的结果")
        major[candidate_id] = bool(candidate_review.get("major_theme_issue"))

    viable = [
        item
        for item in ordered
        if not major[str(item["candidate_id"])]
    ]
    if len(viable) == 1:
        return viable[0], "major_theme_issue_filter"

    pool = viable if viable else ordered
    subjective_scores = {
        str(item["candidate_id"]): float(
            reviews[str(item["candidate_id"])]["weighted_total"]
        )
        for item in pool
    }
    score_gap = abs(
        subjective_scores[str(pool[0]["candidate_id"])]
        - subjective_scores[str(pool[1]["candidate_id"])]
    )
    if score_gap >= SUBJECTIVE_NEAR_TIE_POINTS:
        return (
            max(
                pool,
                key=lambda item: subjective_scores[str(item["candidate_id"])],
            ),
            "subjective_total",
        )
    return max(pool, key=rule_rank), "near_tie_rule_tiebreak"


class LyricJudgeAgent:
    """在不知道 persona 和规则分的前提下评价两份歌词。"""

    def __init__(self, llm: JsonLLM) -> None:
        self._llm = llm

    def run(
        self,
        requirement: SongRequirement,
        candidates: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        if len(candidates) != 2:
            raise ValueError("主观评审必须收到两个候选")

        ordered = sorted(candidates, key=lambda item: str(item["candidate_id"]))
        seed = "|".join(
            [requirement.theme, *(str(item["candidate_id"]) for item in ordered)]
        )
        if hashlib.sha256(seed.encode("utf-8")).digest()[0] % 2:
            ordered.reverse()
        labels = {"A": ordered[0], "B": ordered[1]}

        def anonymous_text(label: str) -> str:
            draft = LyricDraft.model_validate(labels[label]["draft"])
            return f"[{label}]\n标题：{draft.title}\nHook：{draft.hook}\n歌词：\n{draft.lyrics}"

        payload = self._llm.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是中文流行歌词的匿名评审。只根据歌曲需求和匿名 A/B 歌词判断，"
                        "不得猜测作者身份，不得使用押韵规则分或任何外部信息。必须只输出 JSON 对象。"
                        "分别按 1 到 5 分、0.5 分档评价：主题契合、自然度、情绪控制、"
                        "叙事与意象一致性、Hook 记忆点。每一项必须引用具体歌词证据并解释理由。"
                        "统一评分锚点：5 分表示完整、准确且有鲜明细节，4 分表示整体成立但有轻微"
                        "瑕疵，3 分表示基本可用但存在明显模板化、缺层或矛盾，2 分表示关键要求"
                        "明显缺失或偏离，1 分表示核心命题错误。不得因为辞藻华丽或押韵整齐自动"
                        "给高分。主题契合要检查叙事方向、人物关系和场景是否与需求一致；自然度"
                        "要区分真实生活细节与套话、口号、为修辞而修辞；情绪控制同时检查情绪"
                        "强度和需求中每一层情绪是否完整，要求多种情绪交织时，遗漏一层不能给"
                        "高分；叙事与意象一致性要检查前后动作、时间、空间及比喻转换是否有可"
                        "理解的路径；Hook 记忆点要同时考虑辨识度、主题承载和可复唱性。必须结合"
                        "整段上下文判断，不能只凭一个命令句、否定句或比喻作表面结论。"
                        "问题 code 只能是 theme_drift、emotion_overreach、emotion_incomplete、"
                        "generic_expression、imagery_mismatch、narrative_gap、hook_weakness、other；"
                        "severity 只能是 minor 或 major。主题方向反转、人物关系误写或把克制情绪"
                        "推向生死悲剧，应标为 major theme_drift。"
                        "JSON 输出结构示例（仅说明结构，不得照抄内容）："
                        '{"candidate_a":{"theme_fit":{"score":4.5,"evidence":"某句",'
                        '"reason":"理由"},"naturalness":{"score":4.0,"evidence":"某句",'
                        '"reason":"理由"},"emotional_control":{"score":4.0,"evidence":"某句",'
                        '"reason":"理由"},"narrative_imagery_coherence":{"score":4.0,'
                        '"evidence":"某句","reason":"理由"},"hook_memorability":{"score":4.0,'
                        '"evidence":"某句","reason":"理由"},"strengths":["优点"],'
                        '"issues":[{"code":"theme_drift","severity":"major",'
                        '"evidence":"某句","suggestion":"建议"}]},"candidate_b":'
                        '{"theme_fit":{"score":4.0,"evidence":"某句","reason":"理由"},'
                        '"naturalness":{"score":4.0,"evidence":"某句","reason":"理由"},'
                        '"emotional_control":{"score":4.0,"evidence":"某句","reason":"理由"},'
                        '"narrative_imagery_coherence":{"score":4.0,"evidence":"某句",'
                        '"reason":"理由"},"hook_memorability":{"score":4.0,"evidence":"某句",'
                        '"reason":"理由"},"strengths":["优点"],"issues":[]},'
                        '"recommended_winner":"A","confidence":"medium",'
                        '"comparison_summary":"比较结论"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "歌曲需求：\n"
                        f"{json.dumps(requirement.model_dump(), ensure_ascii=False)}\n\n"
                        f"{anonymous_text('A')}\n\n{anonymous_text('B')}\n\n"
                        "请严格按指定 JSON 结构输出匿名比较。"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=2400,
            operation="lyric_judge_initial",
            payload_validator=lambda item: pydantic_validation_feedback(
                LyricJudgeResponse, item
            ),
        )
        response = LyricJudgeResponse.model_validate(payload)
        anonymous_reviews = {
            "A": response.candidate_a,
            "B": response.candidate_b,
        }
        mapping = {
            label: str(candidate["candidate_id"])
            for label, candidate in labels.items()
        }
        reviews_by_candidate: dict[str, dict[str, object]] = {}
        for label, review in anonymous_reviews.items():
            candidate_id = mapping[label]
            review_data = review.model_dump(mode="json")
            review_data["weighted_total"] = weighted_subjective_score(review)
            review_data["major_theme_issue"] = has_major_theme_issue(review)
            review_data["anonymous_label"] = label
            reviews_by_candidate[candidate_id] = review_data

        return {
            "method": "blind_subjective_judge_v1",
            "weights": dict(SUBJECTIVE_WEIGHTS),
            "near_tie_points": SUBJECTIVE_NEAR_TIE_POINTS,
            "anonymous_mapping": mapping,
            "candidate_reviews": reviews_by_candidate,
            "model_recommended_winner": response.recommended_winner,
            "confidence": response.confidence,
            "comparison_summary": response.comparison_summary,
            "both_have_major_theme_issues": all(
                bool(review["major_theme_issue"])
                for review in reviews_by_candidate.values()
            ),
        }
