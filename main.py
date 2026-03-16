import json
import sys
from pathlib import Path

from videocuter.audio_extractor import extract_audio
from videocuter.transcriber import transcribe_audio
from videocuter.analyzer import analyze_transcript
from videocuter.video_cutter import cut_videos

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"请先配置 {CONFIG_PATH}，填入 API key 等信息")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("用法: python main.py <视频文件路径> [输出目录]")
        print("示例: python main.py /path/to/video.mp4 /path/to/output")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(video_path).exists():
        print(f"视频文件不存在: {video_path}")
        sys.exit(1)

    config = load_config()

    print("=" * 60)
    print("步骤 1/4: 提取音频")
    print("=" * 60)
    audio_path = extract_audio(video_path, output_dir)

    print("\n" + "=" * 60)
    print("步骤 2/4: 语音转文字")
    print("=" * 60)
    transcript = transcribe_audio(
        audio_path,
        model_size=config.get("whisper_model", "large-v3"),
        language=config.get("language", "zh"),
    )

    transcript_file = Path(audio_path).with_suffix(".json")
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    print(f"转录结果已保存: {transcript_file}")

    print("\n" + "=" * 60)
    print("步骤 3/4: AI 分析有价值片段")
    print("=" * 60)
    analysis = analyze_transcript(
        transcript,
        api_key=config.get("api_key", ""),
        base_url=config["base_url"],
        model=config.get("model", "deepseek-chat"),
        use_json_format=config.get("use_json_format", True),
    )

    analysis_file = Path(audio_path).with_suffix(".analysis.json")
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存: {analysis_file}")
    print(f"找到 {len(analysis['clips'])} 个有价值片段")

    for clip in analysis["clips"]:
        print(f"  [{clip['start']}s -> {clip['end']}s] {clip['title']}")

    if not analysis["clips"]:
        print("没有找到有价值的片段，程序结束。")
        return

    print("\n" + "=" * 60)
    print("步骤 4/4: 视频切片")
    print("=" * 60)
    clip_dir = output_dir if output_dir else str(Path(video_path).parent / "clips")
    output_files = cut_videos(video_path, analysis["clips"], clip_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"共生成 {len(output_files)} 个视频片段:")
    for f in output_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
