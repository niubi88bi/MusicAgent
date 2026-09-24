"""Build eight new natural Chinese requests for the instrument-pair shadow audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.schemas import LyricDraft, SongRequirement
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_OUTPUT = Path(
    "data/miaoxiang/producer_constraint_instrument_pair_natural.v1.json"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PairNaturalCase(_StrictModel):
    case_id: str = Field(min_length=1)
    risk_focus: str = Field(min_length=1)
    user_request_text: str = Field(min_length=10)
    requirement: SongRequirement
    lyric_draft: LyricDraft


class PairNaturalManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: str = "producer_constraint_instrument_pair_natural"
    frozen_at: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)
    cases: list[PairNaturalCase] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def unique_ids(self) -> "PairNaturalManifest":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("pair_natural_case_ids_must_be_unique")
        return self


def _lyrics(title: str, hook: str, imagery: str) -> dict[str, str]:
    return {
        "title": title,
        "lyrics": (
            f"[Verse]\n{imagery}的灯光落在窗边\n我把今天的脚步放慢一点\n"
            f"旧物和风声陪我走过街面\n把没说完的话收进夜色里面\n\n"
            f"[Chorus]\n{hook}\n让这一刻的呼吸慢慢靠岸\n"
            f"{imagery}仍在远处保持温暖\n我把明天交给第一束光线\n\n"
            f"[Bridge]\n雨声从屋檐退到更远的地方\n我在安静里听见新的回响"
        ),
        "hook": hook,
        "creative_note": "只用于成对乐器自然影子审计，不进行修复或音频生成。",
    }


def _case(
    case_id: str,
    risk_focus: str,
    request: str,
    *,
    theme: str,
    emotion: str,
    genre: str,
    tempo: str,
    vocal: str,
    bpm: int,
    keywords: list[str],
    title: str,
    hook: str,
    imagery: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "risk_focus": risk_focus,
        "user_request_text": request,
        "requirement": {
            "theme": theme,
            "emotion": emotion,
            "genre": genre,
            "tempo": tempo,
            "duration_seconds": 120,
            "perspective": "第一人称",
            "vocal_type": vocal,
            "bpm": bpm,
            "emotion_intensity": 3,
            "keywords": keywords,
            "target_rhyme_group": None,
        },
        "lyric_draft": _lyrics(title, hook, imagery),
    }


def build_manifest() -> dict[str, Any]:
    cases = [
        _case(
            "ns01-bus-depot-guitar",
            "paired_unique_substitution",
            "做一首Sad Slap House中文歌，凌晨公交车库管理员收好最后一串钥匙；不要钢琴，必须用木吉他承担伤感动机，电子推动克制",
            theme="公交车库管理员收好最后一串钥匙",
            emotion="疲惫、克制、告别",
            genre="Sad Slap House",
            tempo="中速克制舞曲",
            vocal="中文独唱，近距离克制表达",
            bpm=124,
            keywords=["公交车库", "钥匙", "凌晨"],
            title="最后一串钥匙",
            hook="最后一串钥匙替我锁住今天",
            imagery="车库",
        ),
        _case(
            "ns02-flower-stall-guitar",
            "paired_unique_substitution",
            "做一首Vina House中文歌，雨后花摊老板收起湿掉的招牌；钢琴全部拿掉，换成原声吉他做轻微伤感动机，整体明亮轻盈",
            theme="雨后花摊老板收起湿掉的招牌",
            emotion="明亮、遗憾、轻盈",
            genre="Vina House",
            tempo="高速轻盈律动",
            vocal="中文独唱，清亮自然",
            bpm=136,
            keywords=["花摊", "雨后", "招牌"],
            title="湿掉的招牌",
            hook="湿掉的招牌也等到了天晴",
            imagery="花摊",
        ),
        _case(
            "ns03-rooftop-guitar",
            "paired_missing_required",
            "做一首Deep House中文歌，屋顶维修工收起最后一卷防水布；必须有木吉他作为伤感动机，不要任何piano，保持温暖克制",
            theme="屋顶维修工收起最后一卷防水布",
            emotion="温暖、疲惫、克制",
            genre="Deep House",
            tempo="中速克制四拍",
            vocal="中文独唱，近麦自然",
            bpm=122,
            keywords=["屋顶", "防水布", "维修工"],
            title="最后一卷防水布",
            hook="最后一卷防水布挡住夜里的风",
            imagery="屋顶",
        ),
        _case(
            "ns04-clinic-guitar",
            "paired_unique_substitution",
            "做一首Sad Slap House中文歌，夜班药房员工整理同事留下的保温杯；禁止使用钢琴，改用木吉他做伤感动机，副歌不要爆发",
            theme="夜班药房员工整理同事留下的保温杯",
            emotion="安静、不舍、接受离开",
            genre="Sad Slap House",
            tempo="中速克制推进",
            vocal="中文独唱，近距离内敛",
            bpm=122,
            keywords=["药房", "保温杯", "夜班"],
            title="留下的保温杯",
            hook="留下的保温杯还温着一整夜",
            imagery="药房",
        ),
        _case(
            "ns05-night-market-compliant",
            "paired_compliant",
            "做一首中文Sad Slap House，夜市收摊后整理旧围巾；不要钢琴，木吉他承担伤感动机，电子推动保持克制",
            theme="夜市收摊后整理旧围巾",
            emotion="克制、怀念、放下",
            genre="Sad Slap House",
            tempo="中速克制舞曲",
            vocal="中文独唱，近距离表达",
            bpm=123,
            keywords=["夜市", "围巾", "收摊"],
            title="收摊后的围巾",
            hook="收摊后的围巾还留着你的温度",
            imagery="夜市",
        ),
        _case(
            "ns06-no-piano-only",
            "forbidden_only",
            "做一首Deep House中文歌，清晨渡口工作人员收起最后一盏灯；不要钢琴，保持温暖克制，不指定替代乐器",
            theme="清晨渡口工作人员收起最后一盏灯",
            emotion="安静、温暖、释然",
            genre="Deep House",
            tempo="中速柔和四拍",
            vocal="中文独唱，自然克制",
            bpm=121,
            keywords=["渡口", "灯", "清晨"],
            title="最后一盏灯",
            hook="最后一盏灯陪我等到天亮",
            imagery="渡口",
        ),
        _case(
            "ns07-recording-ambiguity",
            "paired_ambiguous_context",
            "做一首Sad Slap House中文歌，不要钢琴，必须用木吉他做伤感动机；钢琴只保留为旧录音噪声的反例，不得成为演奏",
            theme="录音室整理旧磁带的夜班工作人员",
            emotion="克制、怀旧、疏离",
            genre="Sad Slap House",
            tempo="中速克制舞曲",
            vocal="中文独唱，近距离表达",
            bpm=124,
            keywords=["录音室", "磁带", "旧录音"],
            title="旧磁带的噪声",
            hook="旧磁带的噪声替我留住昨天",
            imagery="录音室",
        ),
        _case(
            "ns08-warehouse-guitar",
            "paired_missing_required",
            "做一首Vina House中文歌，凌晨仓库管理员关掉最后一排灯；不要piano，acoustic guitar必须承担轻微伤感动机，律动明亮轻盈",
            theme="凌晨仓库管理员关掉最后一排灯",
            emotion="明亮、疲惫、轻盈",
            genre="Vina House",
            tempo="高速但不煽情",
            vocal="中文独唱，清亮自然",
            bpm=138,
            keywords=["仓库", "灯", "凌晨"],
            title="最后一排灯",
            hook="最后一排灯也会等到天亮",
            imagery="仓库",
        ),
    ]
    payload = {
        "schema_version": "1.0",
        "kind": "producer_constraint_instrument_pair_natural",
        "frozen_at": "2026-09-24",
        "notes": [
            "8个新中文自然请求，未复用na09/na10或成对乐器契约holdout。",
            "覆盖唯一替代、缺失必需乐器、已合规、仅禁止钢琴和语义歧义。",
            "每案仅调用一次Producer；影子补丁可应用但不接正式路由，不调用修复模型、不生成音频。",
        ],
        "cases": cases,
    }
    PairNaturalManifest.model_validate(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="生成成对乐器自然影子审计清单。")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.out), "case_count": len(payload["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
