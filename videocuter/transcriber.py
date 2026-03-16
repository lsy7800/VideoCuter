from faster_whisper import WhisperModel


def transcribe_audio(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = "zh",
) -> list[dict]:
    """将音频文件转录为带时间戳的文字列表。

    返回格式: [{"start": 0.0, "end": 5.2, "text": "..."}, ...]
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

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

    return results
