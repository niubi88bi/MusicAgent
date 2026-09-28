"""隔离验证约 120 秒完整歌曲：先做文本准入，再调用 HeartMuLa。

本脚本不接入 Web 默认路径，也不修改现有 30 秒试听行为。文本阶段复用现有
Requirement/Lyricist/评审/规则质检/Producer 图；额外的时长适配检查只作为本次
验证的准入条件，不改变正式评分或回炉路由。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.critic_agent import LyricJudgeAgent
from agents.graph import build_music_graph
from agents.lyricist_agent import LYRICIST_PERSONAS
from agents.producer_agent import ProducerAgent
from agents.requirement_agent import RequirementAgent
from agents.schemas import (
    JsonLLM,
    LyricDraft,
    ProductionPlan,
    SongRequirement,
    pydantic_validation_feedback,
)
from services.llm import DeepSeekLLM
from services.music_gen import HeartMuLaGenerator
from services.workflow_artifacts import write_text_artifacts
from tools.lyrics_quality.parser import parse_lyric_lines
from tools.lyrics_quality.scorer import score_lyrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/output/full-song-120-validation"
LABEL_PATTERN = re.compile(r"^\[([^\]]+)\]$")
UNFINISHED_MARKERS = ("待续", "未完", "省略", "...", "……")
MAX_DURATION_FIT_REWRITES = 2
RETEST_PRONUNCIATION_SOURCE = "照亮脚下的坑洼"
RETEST_PRONUNCIATION_REPLACEMENT = "照亮轻快的步伐"
RETEST_EXPECTED_PRONUNCIATION_OCCURRENCES = 3
RETEST_TERMINAL_VERSE = """[Verse]
推开家门换鞋把外套挂架
窗台绿萝安静等我说话
晚风从窗缝溜进来道晚安
我笑着说明天还要一起走啊"""


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _count_chinese(text: str) -> int:
    return sum("\u4e00" <= char <= "\u9fff" for char in text)


def build_manual_feedback_retest_lyrics(lyrics: str) -> tuple[str, dict[str, Any]]:
    """只应用人工试听确认的两项唯一修订；无法唯一匹配时失败关闭。"""

    normalized = lyrics.rstrip()
    pronunciation_occurrences = normalized.count(RETEST_PRONUNCIATION_SOURCE)
    terminal_verse_occurrences = normalized.count(RETEST_TERMINAL_VERSE)
    if pronunciation_occurrences != RETEST_EXPECTED_PRONUNCIATION_OCCURRENCES:
        raise ValueError(
            "发音修订片段数量不符合冻结预期："
            f"expected={RETEST_EXPECTED_PRONUNCIATION_OCCURRENCES}, "
            f"actual={pronunciation_occurrences}"
        )
    if terminal_verse_occurrences != 1 or not normalized.endswith(
        RETEST_TERMINAL_VERSE
    ):
        raise ValueError(
            "末尾 Verse 不是唯一且位于歌词结尾，拒绝非确定性删除："
            f"occurrences={terminal_verse_occurrences}"
        )

    without_terminal_verse = normalized[: -len(RETEST_TERMINAL_VERSE)].rstrip()
    revised = without_terminal_verse.replace(
        RETEST_PRONUNCIATION_SOURCE,
        RETEST_PRONUNCIATION_REPLACEMENT,
    )
    return revised, {
        "patch_mode": "deterministic_manual_feedback_retest_v1",
        "pronunciation_patch": {
            "source": RETEST_PRONUNCIATION_SOURCE,
            "replacement": RETEST_PRONUNCIATION_REPLACEMENT,
            "occurrences": pronunciation_occurrences,
        },
        "removed_terminal_block": RETEST_TERMINAL_VERSE,
        "removed_terminal_block_occurrences": terminal_verse_occurrences,
    }


def _section_blocks(lyrics: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in lyrics.splitlines():
        line = raw_line.strip()
        match = LABEL_PATTERN.fullmatch(line)
        if match:
            if current is not None:
                blocks.append(current)
            current = {"label": match.group(1), "lines": []}
        elif line and current is not None:
            current["lines"].append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def audit_duration_fit(
    lyrics: str,
    hook: str,
    *,
    duration_seconds: int,
) -> dict[str, Any]:
    """检查长歌候选的基本完整性，不把生成安全上限写入正式质量分。"""

    blocks = _section_blocks(lyrics)
    parsed_lines = parse_lyric_lines(lyrics)
    supported_labels = [block["label"].title() for block in blocks]
    supported_counts = Counter(line.section for line in parsed_lines)
    singing_lines = [
        line
        for block in blocks
        for line in block["lines"]
        if _count_chinese(line) > 0
    ]
    line_texts = [line.strip() for line in singing_lines]
    duplicate_line_count = len(line_texts) - len(set(line_texts))
    chorus_blocks = [
        block
        for block in blocks
        if block["label"].strip().lower().startswith("chorus")
    ]
    verse_blocks = [
        block
        for block in blocks
        if block["label"].strip().lower().startswith("verse")
    ]
    bridge_blocks = [
        block
        for block in blocks
        if block["label"].strip().lower().startswith("bridge")
    ]
    hook_occurrences = sum(
        1 for block in chorus_blocks for line in block["lines"] if hook.strip() == line
    )
    has_unfinished_marker = any(marker in lyrics for marker in UNFINISHED_MARKERS)
    issues: list[str] = []

    # 这些阈值是本次隔离验证的保守 sanity check，不随生成安全上限线性增加。
    if len(singing_lines) < 24:
        issues.append(f"演唱行仅 {len(singing_lines)} 行，低于本次验证下限 24 行")
    chinese_chars = _count_chinese("\n".join(line_texts))
    if chinese_chars < 260:
        issues.append(f"中文总字数仅 {chinese_chars}，低于本次验证下限 260 字")
    if len(verse_blocks) < 2:
        issues.append("至少需要两段主歌")
    if len(chorus_blocks) < 2:
        issues.append("至少需要两段副歌")
    if not bridge_blocks:
        issues.append("缺少 Bridge；本次用 Bridge 表示预副歌/尾部过渡")
    if any(not block["lines"] for block in blocks):
        issues.append("存在空段落标签")
    if not hook.strip() or hook_occurrences < 2:
        issues.append(f"Hook 在副歌中的完整出现次数为 {hook_occurrences}，低于 2 次")
    if has_unfinished_marker:
        issues.append("正文含有待续、省略或未完成标记")
    if supported_counts.get("Verse", 0) < 8:
        issues.append("现有质检可解析的 Verse 演唱行少于 8 行")
    if supported_counts.get("Chorus", 0) < 8:
        issues.append("现有质检可解析的 Chorus 演唱行少于 8 行")

    return {
        "audit_name": "full_song_duration_fit_shadow_v1",
        "advisory_only_for_formal_quality": True,
        "requested_duration_seconds": duration_seconds,
        "duration_semantics": "generation_upper_bound",
        "passed": not issues,
        "issues": issues,
        "section_labels": supported_labels,
        "section_block_count": len(blocks),
        "section_block_line_counts": [
            {"label": block["label"], "line_count": len(block["lines"])}
            for block in blocks
        ],
        "supported_parser_counts": dict(supported_counts),
        "verse_block_count": len(verse_blocks),
        "chorus_block_count": len(chorus_blocks),
        "bridge_block_count": len(bridge_blocks),
        "singing_line_count": len(singing_lines),
        "chinese_character_count": chinese_chars,
        "unique_singing_line_count": len(set(line_texts)),
        "duplicate_singing_line_count": duplicate_line_count,
        "duplicate_line_ratio": round(
            duplicate_line_count / len(line_texts), 4
        )
        if line_texts
        else None,
        "hook": hook,
        "hook_occurrences_in_chorus": hook_occurrences,
        "has_unfinished_marker": has_unfinished_marker,
        "structure_mapping": (
            "现有质检只解析 Verse/Chorus/Bridge；本次 Bridge 可承载预副歌或尾部过渡，"
            "前奏/尾奏如无演唱行不计入演唱行。"
        ),
    }


class FullSongLyricistAgent:
    """只供本次隔离验证使用的长歌提示，复用正式歌词契约。"""

    def __init__(self, llm: JsonLLM, persona: str) -> None:
        if persona not in LYRICIST_PERSONAS:
            raise ValueError(f"未知作词人 persona：{persona}")
        self._llm = llm
        self._persona = LYRICIST_PERSONAS[persona]

    @property
    def persona_id(self) -> str:
        return self._persona.persona_id

    @property
    def display_name(self) -> str:
        return self._persona.display_name

    def run(
        self,
        requirement: SongRequirement,
        revision_feedback: str | None = None,
        previous_draft: LyricDraft | None = None,
    ) -> LyricDraft:
        feedback = revision_feedback or "无"
        previous_lyrics = "无（这是首次创作）"
        revision_instruction = "请创作一版适配约 120 秒演唱的完整歌词。"
        if previous_draft is not None:
            previous_lyrics = (
                f"标题：{previous_draft.title}\n"
                f"Hook：{previous_draft.hook}\n"
                f"歌词：\n{previous_draft.lyrics}"
            )
            revision_instruction = (
                "这是回炉修改：以上一版为基础，只修复反馈问题并补足必要内容，"
                "保留主题、Hook、自然意象和未涉及的有效句子。"
            )
        rhyme_instruction = (
            f"需求中的目标辙口是「{requirement.target_rhyme_group}」。这是现有规则质检的正式约束："
            "每个 Verse、Chorus、Bridge 的多数演唱行都要使用该辙口的自然句尾；"
            "不要为了凑字数改成跨辙硬押，也不要只让副歌押韵而让主歌完全散掉。"
            if requirement.target_rhyme_group
            else "需求没有指定目标辙口，请保持各段句尾自然且有一致性。"
        )
        payload = self._llm.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是擅长中文流行音乐的作词人。必须只输出 JSON 对象，不要 Markdown。"
                        "lyrics 必须使用英文方括号段落标签，至少两段 [Verse]、两段 [Chorus]、"
                        "一段 [Bridge]；只使用 Verse、Chorus、Bridge 这三类标签。"
                        "这是约 120 秒完整歌曲验证，不是 30 秒试听：正文至少 24 行、260 个中文字符，"
                        "每个主歌和副歌至少 4 行；副歌至少完整出现两次 Hook。"
                        "可以用 Bridge 承载预副歌或尾部过渡，前奏/尾奏若无演唱行可不写。"
                        "不要写省略号、待续或未完成占位内容。hook 是一句短而好记的中文歌词，"
                        "必须在至少两段 Chorus 中原样出现。避免真实歌手名和侵权歌词。"
                        f"{rhyme_instruction}"
                        f"你的创作侧重点：{self._persona.instruction}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "歌曲需求：\n"
                        f"{json.dumps(requirement.model_dump(), ensure_ascii=False)}\n\n"
                        f"上一版歌词：\n{previous_lyrics}\n\n"
                        f"修改反馈：{feedback}\n\n"
                        f"{revision_instruction}\n"
                        "请输出 JSON，字段必须为：title、lyrics、hook、creative_note。"
                    ),
                },
            ],
            temperature=0.85,
            max_tokens=3000,
            operation=self._operation_name(revision_feedback is not None),
            payload_validator=lambda payload: pydantic_validation_feedback(
                LyricDraft, payload
            ),
        )
        return LyricDraft.model_validate(payload)

    def _operation_name(self, is_rewrite: bool) -> str:
        suffix = "rewrite" if is_rewrite else "initial"
        return f"full_song_{self.persona_id}_{suffix}"


def _build_text_graph(llm: DeepSeekLLM):
    return build_music_graph(
        RequirementAgent(llm),
        FullSongLyricistAgent(llm, persona="structured"),
        ProducerAgent(llm),
        secondary_lyricist_agent=FullSongLyricistAgent(llm, persona="imagery"),
        lyric_judge_agent=LyricJudgeAgent(llm),
        text_only=True,
    )


def _quality_report_for_draft(
    draft: LyricDraft,
    requirement: SongRequirement,
    *,
    duration_seconds: int,
) -> dict[str, Any]:
    report = score_lyrics(
        lyrics=draft.lyrics,
        hook=draft.hook,
        duration_seconds=duration_seconds,
        target_rhyme_group=requirement.target_rhyme_group,
        theme=requirement.theme,
        theme_keywords=requirement.keywords,
    )
    return report.to_dict()


def _duration_fit_feedback(audit: dict[str, Any]) -> str:
    issues = "；".join(str(item) for item in audit.get("issues", [])) or "演唱文本偏短"
    return (
        "这是 120 秒隔离验证的时长适配回炉，不是自由另起一稿。请以上一版为基础，"
        "保留主题、Hook、已有自然意象和可用韵脚，只补足完整演唱内容。"
        "要求：至少两段主歌、至少两段副歌和一段 Bridge；总演唱行不少于 24 行、"
        "中文字符不少于 260；副歌至少完整重复 Hook 两次；不要使用‘待续’、省略号或"
        f"未完成标记。当前额外审计问题：{issues}。只输出契约要求的 JSON。"
    )


def run_text_stage(theme: str, duration_seconds: int, job_dir: Path) -> bool:
    started_at = time.perf_counter()
    llm = DeepSeekLLM()
    final_state: dict[str, object] = {}
    try:
        graph = _build_text_graph(llm)
        for update in graph.stream(
            {
                "theme": theme,
                "requested_duration_seconds": duration_seconds,
            },
            stream_mode="updates",
        ):
            for node_name, changes in update.items():
                final_state.update(changes)
                print(f"文本节点完成：{node_name}", flush=True)

        requirement = final_state.get("requirement")
        draft = final_state.get("lyric_draft")
        plan = final_state.get("production_plan")
        if not isinstance(requirement, SongRequirement):
            raise TypeError("文本图未返回有效 SongRequirement")
        if not isinstance(draft, LyricDraft):
            raise TypeError("文本图未返回有效 LyricDraft")
        if not isinstance(plan, ProductionPlan):
            raise TypeError("文本图未返回有效 ProductionPlan")

        quality_report = final_state.get("quality_report")
        if not isinstance(quality_report, dict):
            raise TypeError("文本图未返回有效 quality_report")
        duration_audit = audit_duration_fit(
            draft.lyrics,
            draft.hook,
            duration_seconds=duration_seconds,
        )
        duration_fit_rewrites: list[dict[str, Any]] = []
        duration_fit_rewrite_used = False
        quality_ok = (
            final_state.get("quality_status") == "passed"
            and quality_report.get("passed") is True
        )

        # 只有现有正式质检已经通过、但额外完整性审计不足时，才允许在本次
        # 隔离任务中使用现有 LyricistAgent 的回炉接口补写；不改正式图的路由。
        if quality_ok and not duration_audit["passed"]:
            selected_persona = str(final_state.get("selected_lyricist") or "structured")
            if selected_persona not in {"structured", "imagery"}:
                selected_persona = "structured"
            rewrite_agent = FullSongLyricistAgent(llm, persona=selected_persona)
            previous_draft = draft
            for rewrite_index in range(1, MAX_DURATION_FIT_REWRITES + 1):
                revision_feedback = _duration_fit_feedback(duration_audit)
                rewritten = rewrite_agent.run(
                    requirement,
                    revision_feedback=revision_feedback,
                    previous_draft=previous_draft,
                )
                rewritten_report = _quality_report_for_draft(
                    rewritten,
                    requirement,
                    duration_seconds=duration_seconds,
                )
                rewritten_audit = audit_duration_fit(
                    rewritten.lyrics,
                    rewritten.hook,
                    duration_seconds=duration_seconds,
                )
                duration_fit_rewrites.append(
                    {
                        "rewrite_index": rewrite_index,
                        "persona": selected_persona,
                        "revision_feedback": revision_feedback,
                        "draft": rewritten.model_dump(mode="json"),
                        "quality_report": rewritten_report,
                        "duration_fit_audit": rewritten_audit,
                    }
                )
                if rewritten_report.get("passed") is True and rewritten_audit["passed"]:
                    final_state.update(
                        {
                            "lyric_draft": rewritten,
                            "lyrics": rewritten.lyrics,
                            "quality_report": rewritten_report,
                            "scores": {
                                "rhyme": rewritten_report["rhyme"]["score"],
                                "structure": rewritten_report["structure"]["score"],
                                "total": rewritten_report["total_score"],
                            },
                            "quality_status": "passed",
                            "attempts": int(final_state.get("attempts", 1)) + rewrite_index,
                            "selected_attempt": int(final_state.get("attempts", 1)) + rewrite_index,
                        }
                    )
                    previous_draft = rewritten
                    draft = rewritten
                    quality_report = rewritten_report
                    duration_audit = rewritten_audit
                    duration_fit_rewrite_used = True
                    history = final_state.get("quality_history")
                    if isinstance(history, list):
                        history.append(
                            {
                                "attempt": final_state["attempts"],
                                "title": rewritten.title,
                                "lyrics": rewritten.lyrics,
                                "hook": rewritten.hook,
                                "scores": final_state["scores"],
                                "quality_report": rewritten_report,
                                "quality_status": "passed",
                                "revision_feedback": revision_feedback,
                                "revision_source_attempt": final_state["attempts"] - 1,
                                "duration_fit_shadow_rewrite": True,
                            }
                        )
                    break
                previous_draft = rewritten

        write_text_artifacts(job_dir, final_state)
        _write_json(job_dir / "duration_fit_audit.json", duration_audit)
        _write_json(
            job_dir / "duration_fit_rewrite_attempts.json",
            {
                "max_rewrites": MAX_DURATION_FIT_REWRITES,
                "rewrite_used": duration_fit_rewrite_used,
                "attempts": duration_fit_rewrites,
            },
        )
        _write_json(
            job_dir / "text_stage.json",
            {
                "theme": theme,
                "requested_duration_seconds": duration_seconds,
                "quality_status": final_state.get("quality_status"),
                "quality_passed": bool(quality_report.get("passed")),
                "duration_fit_passed": duration_audit["passed"],
                "duration_fit_rewrite_used": duration_fit_rewrite_used,
                "duration_fit_rewrite_count": len(duration_fit_rewrites),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "llm_telemetry": DeepSeekLLM.summarize_call_records(
                    llm.telemetry_snapshot()
                ),
            },
        )
        quality_ok = (
            final_state.get("quality_status") == "passed"
            and quality_report.get("passed") is True
        )
        admitted = quality_ok and duration_audit["passed"] is True
        print(
            "文本准入："
            f"quality_status={final_state.get('quality_status')}, "
            f"quality_passed={quality_ok}, "
            f"duration_fit_passed={duration_audit['passed']}, admitted={admitted}",
            flush=True,
        )
        if not admitted:
            print(
                "未进入音频阶段；文本、正式质检和额外审计产物已保留。",
                flush=True,
            )
        return admitted
    except Exception as error:
        _write_json(
            job_dir / "failure.json",
            {
                "stage": "text",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "llm_telemetry": DeepSeekLLM.summarize_call_records(
                    llm.telemetry_snapshot()
                ),
            },
        )
        print(f"文本阶段失败：{type(error).__name__}: {error}", file=sys.stderr)
        return False
    finally:
        llm.close()


def prepare_manual_feedback_retest(
    source_job_dir: Path,
    duration_seconds: int,
    output_root: Path,
) -> tuple[Path, bool]:
    """从冻结源任务构造确定性复验文本，不调用 LLM、不覆盖源产物。"""

    started_at = time.perf_counter()
    job_dir = (output_root / uuid4().hex).resolve()
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        requirement = SongRequirement.model_validate_json(
            (source_job_dir / "requirement.json").read_text(encoding="utf-8")
        )
        source_draft = LyricDraft.model_validate_json(
            (source_job_dir / "lyric_draft.json").read_text(encoding="utf-8")
        )
        plan = ProductionPlan.model_validate_json(
            (source_job_dir / "production_plan.json").read_text(encoding="utf-8")
        )
        revised_lyrics, patch_report = build_manual_feedback_retest_lyrics(
            source_draft.lyrics
        )
        revised_note = source_draft.creative_note.replace(
            "Hook重复四次", "Hook重复三次"
        )
        revised_note = (
            f"{revised_note} 人工试听复验版删除末尾Verse，并将易误唱词替换为“步伐”。"
        )
        revised_draft = source_draft.model_copy(
            update={"lyrics": revised_lyrics, "creative_note": revised_note}
        )
        quality_report = _quality_report_for_draft(
            revised_draft,
            requirement,
            duration_seconds=duration_seconds,
        )
        duration_audit = audit_duration_fit(
            revised_draft.lyrics,
            revised_draft.hook,
            duration_seconds=duration_seconds,
        )
        quality_ok = quality_report.get("passed") is True
        admitted = quality_ok and duration_audit.get("passed") is True
        scores = {
            "rhyme": quality_report["rhyme"]["score"],
            "structure": quality_report["structure"]["score"],
            "total": quality_report["total_score"],
        }

        _write_json(
            job_dir / "requirement.json", requirement.model_dump(mode="json")
        )
        _write_json(
            job_dir / "lyric_draft.json", revised_draft.model_dump(mode="json")
        )
        _write_json(job_dir / "production_plan.json", plan.model_dump(mode="json"))
        _write_json(
            job_dir / "quality_report.json",
            {
                "quality_status": "passed" if quality_ok else "capped",
                "attempts": 1,
                "selected_attempt": 1,
                "scores": scores,
                "quality_report": quality_report,
            },
        )
        _write_json(job_dir / "duration_fit_audit.json", duration_audit)
        _write_json(
            job_dir / "retest_preparation.json",
            {
                "source_job_dir": str(source_job_dir.resolve()),
                "requested_duration_seconds": duration_seconds,
                "duration_semantics": "generation_upper_bound",
                "source_manual_review": {
                    "lyrics_complete": False,
                    "missing_line_estimate": "4-5",
                    "pronunciation_issue": "坑洼被唱成坑刷",
                    "end_to_end_status": "failed",
                },
                "patch_report": patch_report,
                "llm_calls": 0,
                "source_job_preserved": True,
            },
        )
        _write_json(
            job_dir / "text_stage.json",
            {
                "theme": requirement.theme,
                "requested_duration_seconds": duration_seconds,
                "duration_semantics": "generation_upper_bound",
                "mode": "deterministic_manual_feedback_retest_v1",
                "quality_status": "passed" if quality_ok else "capped",
                "quality_passed": quality_ok,
                "duration_fit_passed": duration_audit.get("passed") is True,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "llm_telemetry": {
                    "logical_call_count": 0,
                    "api_attempt_count": 0,
                    "total_tokens": 0,
                },
            },
        )
        (job_dir / "lyrics.txt").write_text(
            revised_draft.lyrics + "\n", encoding="utf-8"
        )
        print(f"复验任务目录：{job_dir}", flush=True)
        print(
            "复验文本准入："
            f"quality_passed={quality_ok}, "
            f"duration_fit_passed={duration_audit.get('passed')}, "
            f"admitted={admitted}",
            flush=True,
        )
        return job_dir, admitted
    except Exception as error:
        _write_json(
            job_dir / "failure.json",
            {
                "stage": "retest_text",
                "source_job_dir": str(source_job_dir.resolve()),
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "source_job_preserved": True,
                "llm_calls": 0,
            },
        )
        print(f"复验文本阶段失败：{type(error).__name__}: {error}", file=sys.stderr)
        return job_dir, False


def _run_tool(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        return {
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def probe_audio(path: Path, requested_duration: int) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "file_exists": path.is_file(),
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
        "generation_limit_seconds": requested_duration,
        "ffprobe": None,
        "decode_check": None,
        "duration_close_to_limit": False,
        "inferred_stop_reason": "unknown",
        "manual_listen_required": True,
        "manual_listen_status": "pending",
    }
    if not path.is_file() or path.stat().st_size == 0:
        return result
    if ffprobe:
        probe = _run_tool(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name,size",
                "-of",
                "json",
                str(path),
            ]
        )
        result["ffprobe"] = probe
        if probe["returncode"] == 0:
            try:
                payload = json.loads(probe["stdout"])
                fmt = payload.get("format", {})
                actual = float(fmt["duration"])
                result["actual_duration_seconds"] = round(actual, 3)
                result["limit_detection_tolerance_seconds"] = 0.5
                result["duration_close_to_limit"] = (
                    actual >= requested_duration - 0.5
                )
                result["inferred_stop_reason"] = (
                    "generation_limit_reached"
                    if result["duration_close_to_limit"]
                    else "model_eos_before_limit"
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                result["ffprobe_parse_error"] = str(error)
    else:
        result["ffprobe_unavailable"] = True
    if ffmpeg:
        result["decode_check"] = _run_tool(
            [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"]
        )
        result["decodable"] = result["decode_check"]["returncode"] == 0
    else:
        result["decode_check_unavailable"] = True
    result["automated_audio_checks_passed"] = bool(
        result.get("file_exists")
        and result.get("actual_duration_seconds", 0) > 0
        and result.get("decodable") is True
    )
    return result


def run_audio_stage(job_dir: Path, duration_seconds: int) -> bool:
    started_at = time.perf_counter()
    try:
        quality = json.loads(
            (job_dir / "quality_report.json").read_text(encoding="utf-8")
        )
        duration_audit = json.loads(
            (job_dir / "duration_fit_audit.json").read_text(encoding="utf-8")
        )
        if quality.get("quality_status") != "passed" or not quality.get(
            "quality_report", {}
        ).get("passed"):
            raise RuntimeError("文本正式质检未通过，拒绝启动 HeartMuLa")
        if not duration_audit.get("passed"):
            raise RuntimeError("120 秒歌词额外完整性审计未通过，拒绝启动 HeartMuLa")

        requirement = SongRequirement.model_validate_json(
            (job_dir / "requirement.json").read_text(encoding="utf-8")
        )
        draft = LyricDraft.model_validate_json(
            (job_dir / "lyric_draft.json").read_text(encoding="utf-8")
        )
        plan = ProductionPlan.model_validate_json(
            (job_dir / "production_plan.json").read_text(encoding="utf-8")
        )
        del requirement
        generator = HeartMuLaGenerator.from_settings(output_dir=job_dir / "audio")
        print(f"开始 HeartMuLa 生成，安全上限 {duration_seconds} 秒…", flush=True)
        result = generator.generate(
            lyrics=draft.lyrics,
            tags=plan.tags,
            duration_seconds=duration_seconds,
            job_id="music",
        )
        audio_checks = probe_audio(result.output_path, duration_seconds)
        payload = {
            "status": "done" if audio_checks["automated_audio_checks_passed"] else "needs_review",
            "generation_limit_seconds": duration_seconds,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "audio": audio_checks,
            "note": "停止原因由成品时长相对上限推断；自动检查不能替代人工试听，仍需确认歌词覆盖、发音和结尾。",
        }
        _write_json(job_dir / "audio_result.json", payload)
        print(f"音频产物：{result.output_path.resolve()}", flush=True)
        print(
            "音频自动检查："
            f"duration={audio_checks.get('actual_duration_seconds')}, "
            f"decodable={audio_checks.get('decodable')}, "
            f"automated_passed={audio_checks['automated_audio_checks_passed']}",
            flush=True,
        )
        return bool(audio_checks["automated_audio_checks_passed"])
    except Exception as error:
        _write_json(
            job_dir / "audio_result.json",
            {
                "status": "failed",
                "requested_duration_seconds": duration_seconds,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "error_type": type(error).__name__,
                "error": str(error),
                "text_artifacts_preserved": True,
            },
        )
        print(f"音频阶段失败：{type(error).__name__}: {error}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="隔离验证长歌自然结束与歌词覆盖")
    parser.add_argument(
        "--stage",
        choices=("text", "audio", "all", "retest-text", "retest-all"),
        default="all",
    )
    parser.add_argument("--theme", default="晚风陪我回家")
    parser.add_argument("--duration", type=int, default=240)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--job-dir", type=Path, help="audio 阶段使用已有文本产物目录")
    parser.add_argument(
        "--source-job",
        type=Path,
        help="retest 阶段使用的冻结源任务目录",
    )
    args = parser.parse_args()
    if not 120 <= args.duration <= 240:
        parser.error("本隔离脚本只接受 120 到 240 秒的生成安全上限")

    os.chdir(PROJECT_ROOT)
    if args.stage == "audio":
        if args.job_dir is None:
            parser.error("--stage audio 必须提供 --job-dir")
        return 0 if run_audio_stage(args.job_dir.resolve(), args.duration) else 1

    if args.stage in {"retest-text", "retest-all"}:
        if args.source_job is None:
            parser.error("retest 阶段必须提供 --source-job")
        retest_job_dir, text_ok = prepare_manual_feedback_retest(
            args.source_job.resolve(),
            args.duration,
            args.out.resolve(),
        )
        if not text_ok or args.stage == "retest-text":
            return 0 if text_ok else 1
        return 0 if run_audio_stage(retest_job_dir, args.duration) else 1

    job_dir = (args.out / uuid4().hex).resolve()
    job_dir.mkdir(parents=True, exist_ok=False)
    print(f"验证任务目录：{job_dir}", flush=True)
    text_ok = run_text_stage(args.theme, args.duration, job_dir)
    if not text_ok:
        return 2
    if args.stage == "text":
        return 0
    return 0 if run_audio_stage(job_dir, args.duration) else 1


if __name__ == "__main__":
    raise SystemExit(main())
