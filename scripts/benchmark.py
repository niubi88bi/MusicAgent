"""D1 基准测试：验证 4060 跑 HeartMuLa 的速度与显存占用

用法（venv 激活后）:
    python scripts/benchmark.py --ckpt data/ckpt/HeartMuLaGen \
        --out data/output/benchmark

记录项:
- 模型加载耗时 / 显存峰值
- 30s / 60s 音频生成耗时
- 每 10 秒音频的生成耗时（估算 RTF）
"""
import argparse
import json
import os
import subprocess
import time

import torch
from heartlib import HeartMuLaGenPipeline


def nvidia_smi_mem():
    """查询当前 GPU 显存占用（MiB）"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        used, total = out.strip().split(",")
        return int(used), int(total)
    except Exception:
        return -1, -1


def build_lyrics(duration_sec: int) -> str:
    """按目标时长生成一段带结构标签的测试歌词（英文，规避中文发音变量）"""
    verse = (
        "[Verse]\n"
        "The morning light comes through the glass\n"
        "Another day for us to pass\n"
        "I feel the rhythm in my chest\n"
        "The simple moments are the best\n\n"
    )
    chorus = (
        "[Chorus]\n"
        "We sing along to the steady beat\n"
        "The music carries down the street\n"
        "Every step we take is new\n"
        "We sing our song for me and you\n\n"
    )
    if duration_sec <= 45:
        return verse + chorus
    return verse + chorus + verse + chorus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/ckpt/HeartMuLaGen")
    ap.add_argument("--out", default="data/output/benchmark")
    ap.add_argument("--durations", default="30,60")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = {}

    # ---- 1. 加载模型（计时 + 显存） ----
    used0, total = nvidia_smi_mem()
    t0 = time.time()
    print(f"[bench] 加载模型 {args.ckpt} ...")
    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.ckpt,
        device={"mula": torch.device("cuda"), "codec": torch.device("cuda")},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=True,
    )
    load_s = time.time() - t0
    used1, _ = nvidia_smi_mem()
    results["model_load_sec"] = round(load_s, 1)
    results["vram_load_mib"] = used1 - used0 if used0 >= 0 else -1
    results["vram_total_mib"] = total
    print(f"[bench] 加载完成 {load_s:.1f}s | 显存占用 {(used1-used0) if used0>=0 else -1} MiB / {total} MiB")

    # ---- 2. 分段生成测试 ----
    for dur in [int(x) for x in args.durations.split(",")]:
        lyrics = build_lyrics(dur)
        tags_path = os.path.join(args.out, f"tags_{dur}s.txt")
        lyrics_path = os.path.join(args.out, f"lyrics_{dur}s.txt")
        save_path = os.path.join(args.out, f"song_{dur}s.mp3")
        with open(tags_path, "w") as f:
            f.write("pop,acoustic,guitar,soft male vocal")
        with open(lyrics_path, "w") as f:
            f.write(lyrics)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        print(f"[bench] 生成 {dur}s 音频 ...")
        with torch.no_grad():
            pipe(
                {"lyrics": lyrics_path, "tags": tags_path},
                max_audio_length_ms=dur * 1000,
                save_path=save_path,
            )
        gen_s = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        results[f"gen_{dur}s_sec"] = round(gen_s, 1)
        results[f"gen_{dur}s_rtf"] = round(gen_s / dur, 2)
        results[f"gen_{dur}s_vram_peak_mib"] = round(peak, 0)
        print(f"[bench] {dur}s 生成 {gen_s:.1f}s | RTF={gen_s/dur:.2f} | 显存峰值 {peak:.0f} MiB")

    # ---- 3. 结果落盘 ----
    with open(os.path.join(args.out, "benchmark_result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[bench] 结果已保存: {args.out}/benchmark_result.json")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
