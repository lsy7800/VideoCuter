import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_dir: str | None = None) -> str:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if output_dir is None:
        output_dir = str(video.parent)

    audio_path = str(Path(output_dir) / f"{video.stem}.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"音频提取失败: {result.stderr}")

    print(f"音频已提取到: {audio_path}")
    return audio_path
