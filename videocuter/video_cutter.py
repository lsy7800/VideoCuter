import subprocess
from pathlib import Path


def cut_videos(
    video_path: str,
    clips: list[dict],
    output_dir: str | None = None,
) -> list[str]:
    """根据片段信息对视频进行无损切片。"""
    video = Path(video_path)
    if output_dir is None:
        output_dir = str(video.parent / "clips")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_files = []
    for i, clip in enumerate(clips):
        start = float(clip["start"])
        end = float(clip["end"])
        title = clip.get("title", f"clip_{i + 1}")
        safe_title = "".join(c if c.isalnum() or c in "_ -" else "_" for c in title)
        safe_title = safe_title[:50]

        output_path = str(Path(output_dir) / f"{i + 1:03d}_{safe_title}{video.suffix}")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]

        print(f"正在切片 [{start:.1f}s -> {end:.1f}s]: {title}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"  切片失败: {result.stderr[:200]}")
        else:
            output_files.append(output_path)
            print(f"  已保存: {output_path}")

    return output_files
