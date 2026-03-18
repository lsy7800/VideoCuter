import json
import subprocess
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = "zh",
    output_file: str | None = None,
) -> list[dict]:
    """将音频文件转录为带时间戳的文字列表。

    使用子进程执行转录，避免 CUDA 清理时崩溃导致主进程退出。
    返回格式: [{"start": 0.0, "end": 5.2, "text": "..."}, ...]
    """
    if output_file is None:
        output_file = str(Path(audio_path).with_suffix(".transcript.tmp.json"))

    result = subprocess.run(
        [
            sys.executable, "-m", "videocuter.transcriber",
            audio_path, output_file, model_size, language,
        ],
        capture_output=False,
    )

    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            transcript = json.load(f)
        print(f"转录完成，共 {len(transcript)} 个片段")
        return transcript

    raise RuntimeError(
        f"转录失败 (子进程退出码: {result.returncode})，未生成结果文件"
    )


def _run_transcribe(audio_path: str, output_file: str, model_size: str, language: str):
    """在子进程中实际执行转录，结果写入 JSON 文件。"""
    try:
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"使用设备: {device}, 计算精度: {compute_type}")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(audio_path, language=language, beam_size=5)

    print(f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")

    results = []
    for segment in segments:
        entry = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        }
        results.append(entry)
        print(f"[{entry['start']:.2f}s -> {entry['end']:.2f}s] {entry['text']}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"转录结果已写入: {output_file}")


if __name__ == "__main__":
    _run_transcribe(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
