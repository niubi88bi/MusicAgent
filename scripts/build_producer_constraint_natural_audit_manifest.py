"""Build the frozen 12-case natural hard-constraint audit manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_producer_constraint_natural_audit import NaturalAuditManifest


DEFAULT_OUTPUT = Path("data/miaoxiang/producer_constraint_natural_audit.v1.json")


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
        "creative_note": "只用于Producer自然首稿审计，不进行修复或音频生成。",
    }


def _case(
    case_id: str,
    risk_focus: str,
    request: str,
    expected: list[str],
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
        "expected_hard_concepts": expected,
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
        _case("na01-guofeng-male-courtyard", "explicit_vocal_gender", "做一首唯美抒情国风中文歌，旧院管理员在雨后收起晾晒的信；明确用成熟男声独唱，不要女声，编曲保持含蓄", ["male_vocal", "female_vocal"], theme="旧院管理员收起雨后晾晒的信", emotion="怀旧、含蓄、平静", genre="唯美抒情国风", tempo="舒缓留白", vocal="成熟中文男声独唱，禁止女声", bpm=72, keywords=["旧院", "信件", "雨后"], title="雨后收信", hook="把最后一封信收进雨后的天", imagery="院墙") ,
        _case("na02-vina-female-market", "explicit_vocal_gender", "做一首Vina House中文歌，清晨菜市场收摊的摊主骑车回家；必须是清亮女声独唱，不要男声，高速律动保持轻盈", ["female_vocal", "male_vocal"], theme="菜市场摊主清晨收摊回家", emotion="忙碌后的轻松、明亮", genre="Vina House", tempo="高速轻盈舞曲", vocal="中文清亮女声独唱，禁止男声", bpm=138, keywords=["菜市场", "收摊", "单车"], title="收摊以后", hook="收摊以后天就亮了", imagery="菜市场") ,
        _case("na03-deep-house-male-station", "explicit_vocal_gender", "做一首Deep House中文歌，夜班地铁工作人员关掉最后一盏站台灯；明确成熟男声独唱，不要女声，保持克制", ["male_vocal", "female_vocal"], theme="夜班地铁工作人员关掉最后一盏站台灯", emotion="疲惫、克制、释然", genre="Deep House", tempo="中速克制四拍", vocal="成熟中文男声独唱，禁止女声", bpm=122, keywords=["地铁", "站台灯", "夜班"], title="最后一盏灯", hook="最后一盏灯陪我走出站台", imagery="站台") ,
        _case("na04-dnb-no-rap-night-shift", "explicit_no_rap_mc_slogans", "做一首Drum & Bass中文歌，夜班护士在暴雨中送病人家属到门口；女声完整演唱，不要说唱、MC或采样口号，鼓快但情绪温暖", ["rap_or_mc"], theme="夜班护士在暴雨中送家属到门口", emotion="疲惫、温暖、坚定", genre="Drum & Bass", tempo="高速切分但旋律温暖", vocal="中文女声完整演唱，禁止说唱、MC和采样口号", bpm=172, keywords=["夜班", "暴雨", "门口"], title="门口的灯", hook="门口的灯还替我们亮着", imagery="医院") ,
        _case("na05-future-bass-no-rap-observatory", "explicit_no_rap_mc_slogans", "做一首Future Bass中文歌，山顶气象员记录第一场雪；中文独唱，不要Rap、MC或喊话采样，副歌保持明亮辽阔", ["rap_or_mc"], theme="山顶气象员记录第一场雪", emotion="紧张、惊喜、辽阔", genre="Future Bass", tempo="铺垫后副歌释放", vocal="中文独唱，禁止Rap、MC和喊话采样", bpm=148, keywords=["山顶", "气象员", "初雪"], title="第一场雪写进记录", hook="第一场雪写进记录的空白页", imagery="山顶") ,
        _case("na06-vina-no-rap-riverside", "explicit_no_rap_mc_slogans", "做一首Vina House中文歌，河边早餐摊老板在收最后一只保温桶；中文清亮独唱，不要说唱和MC，律动轻快", ["rap_or_mc"], theme="早餐摊老板收起最后一只保温桶", emotion="清晨、轻快、释然", genre="Vina House", tempo="高速轻快律动", vocal="中文清亮独唱，禁止说唱和MC", bpm=136, keywords=["早餐摊", "保温桶", "河边"], title="最后一只保温桶", hook="最后一只保温桶也装好了清晨", imagery="河边") ,
        _case("na07-techno-waterworks", "explicit_instrumental_only", "做一首Techno纯音乐，凌晨供水站的泵组逐台启动；全程无人声，不要口白、人声采样或短促失真人声，只靠机械循环和滤波推进", ["vocal_material"], theme="凌晨供水站泵组逐台启动", emotion="冷静、机械、逐渐苏醒", genre="Techno纯音乐", tempo="稳定机械循环渐变", vocal="全程纯音乐，无人声、口白或人声采样", bpm=130, keywords=["供水站", "泵组", "阀门"], title="泵组启动序列", hook="泵组启动序列对齐第一拍", imagery="供水站") ,
        _case("na08-ambient-rooftop", "explicit_instrumental_only", "做一首禅意国风Ambient，凌晨天台清洁工扫过积水；全程纯器乐、无人声，不要口白或任何人声素材，也不要歌词", ["vocal_material"], theme="凌晨天台清洁工扫过积水", emotion="安静、空灵、留白", genre="禅意国风Ambient", tempo="自由呼吸、极简推进", vocal="全程纯器乐，无歌词、口白或人声素材", bpm=78, keywords=["天台", "积水", "清扫"], title="天台积水的回声", hook="器乐回声穿过天台的积水", imagery="天台") ,
        _case("na09-sad-slap-no-piano", "explicit_instrument_substitution", "做一首Sad Slap House中文歌，深夜便利店员工整理离职同事留下的围裙；明确不要钢琴，用木吉他做伤感动机，电子推动保持克制", ["piano", "acoustic_guitar"], theme="便利店员工整理离职同事留下的围裙", emotion="克制、不舍、接受离开", genre="Sad Slap House", tempo="中速克制舞曲", vocal="中文独唱，近距离克制表达", bpm=123, keywords=["便利店", "围裙", "夜班"], title="最后一条围裙", hook="最后一条围裙我替你叠好", imagery="便利店") ,
        _case("na10-vina-guitar-substitution", "explicit_instrument_substitution", "做一首Vina House中文歌，清晨花店老板收起最后一束未售出的花；不要钢琴，必须用木吉他做轻微伤感动机，整体保持明亮轻盈", ["piano", "acoustic_guitar"], theme="花店老板收起最后一束未售出的花", emotion="明亮中带一点遗憾", genre="Vina House", tempo="高速轻盈但不煽情", vocal="中文独唱，清亮自然", bpm=136, keywords=["花店", "花束", "清晨"], title="最后一束花", hook="最后一束花也有自己的清晨", imagery="花店") ,
        _case("na11-deep-house-flex-bpm", "explicit_flexible_bpm", "做一首Deep House中文歌，清晨渡船收起最后一根缆绳；BPM只作参考，不要写成必须固定数值，速度可随演唱微调", ["bpm"], theme="清晨渡船收起最后一根缆绳", emotion="安静、疲惫、放松", genre="Deep House", tempo="稳定四拍但速度可微调", vocal="中文独唱，呼吸自然", bpm=123, keywords=["渡船", "缆绳", "江面"], title="收起缆绳", hook="收起缆绳天就快亮了", imagery="渡船") ,
        _case("na12-trance-flex-bpm", "explicit_flexible_bpm", "做一首Trance中文歌，凌晨列车穿过无人的盐碱地；BPM只作参考，可随中文演唱调整，不要锁死固定BPM，保持长线铺陈", ["bpm"], theme="凌晨列车穿过无人的盐碱地", emotion="孤独、辽阔、逐渐明亮", genre="Trance", tempo="长线铺陈、速度随演唱微调", vocal="中文独唱，辽阔但不过度", bpm=132, keywords=["列车", "盐碱地", "凌晨"], title="盐碱地尽头", hook="盐碱地尽头有一盏不熄的灯", imagery="列车") ,
    ]
    payload = {
        "schema_version": "1.0",
        "kind": "producer_constraint_natural_audit",
        "frozen_at": "2026-09-24",
        "rubric": "docs/producer-rag-review-rubric-v2.1.md",
        "notes": [
            "12个请求均为新的中文歌曲场景，未复用此前自然审计清单的case_id。",
            "只生成一次Producer首稿，注入RAG关闭，不执行修复，不生成音频。",
            "硬约束类别按3/3/2/2/2分配，结果用于描述自然违规率，不直接形成生产门槛。",
        ],
        "cases": cases,
    }
    NaturalAuditManifest.model_validate(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="生成Producer自然约束审计清单。")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "case_count": len(payload["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
