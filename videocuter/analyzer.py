import json

from openai import OpenAI

SYSTEM_PROMPT = """你是一个直播视频内容分析专家。你的任务是分析直播视频的转录文本，找出其中有价值的片段。

有价值的片段包括但不限于：
- 有趣的故事或段子
- 有价值的知识分享或教学内容
- 精彩的互动或对话
- 有争议或引发讨论的话题
- 情感高潮或感人的时刻
- 重要的产品介绍或推荐

请忽略以下类型的内容：
- 无意义的寒暄和重复的问候
- 长时间的沉默或无关紧要的闲聊
- 纯粹的广告或重复的营销话术

你必须严格按照以下JSON格式返回结果，不要包含任何其他内容：
{
  "clips": [
    {
      "start": "开始时间（秒）",
      "end": "结束时间（秒）",
      "title": "片段标题（简短描述）",
      "reason": "为什么这个片段有价值"
    }
  ]
}

注意：
1. 时间使用秒数（浮点数）
2. 尽量合并相邻的有价值片段，避免切片过碎
3. 每个片段建议时长在30秒到5分钟之间
4. 如果没有有价值的内容，返回空的clips数组"""


def analyze_transcript(
    transcript: list[dict],
    api_key: str,
    base_url: str,
    model: str = "deepseek-chat",
    max_tokens_per_chunk: int = 8000,
    use_json_format: bool = True,
) -> dict:
    """将转录文本发送给大模型分析，返回有价值的片段信息。"""
    client = OpenAI(api_key=api_key or "ollama", base_url=base_url)

    transcript_text = _format_transcript(transcript)

    chunks = _split_transcript(transcript_text, max_tokens_per_chunk)

    all_clips = []
    for i, chunk in enumerate(chunks):
        print(f"正在分析第 {i + 1}/{len(chunks)} 段文本...")
        result = _analyze_chunk(client, chunk, model, use_json_format)
        if result and "clips" in result:
            all_clips.extend(result["clips"])

    merged = _merge_clips(all_clips)
    return {"clips": merged}


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for entry in transcript:
        lines.append(f"[{entry['start']:.2f}s - {entry['end']:.2f}s] {entry['text']}")
    return "\n".join(lines)


def _split_transcript(text: str, max_chars: int) -> list[str]:
    """按字符数分段，在换行符处切分。"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = []
    current_length = 0

    for line in lines:
        if current_length + len(line) + 1 > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(line)
        current_length += len(line) + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _analyze_chunk(client: OpenAI, chunk: str, model: str, use_json_format: bool = True) -> dict | None:
    user_message = f"请分析以下直播转录文本，找出有价值的片段：\n\n{chunk}"

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
    }
    if use_json_format:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content
    if not content:
        return None

    if not use_json_format:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            content = content[start:end]

    return json.loads(content)


def _merge_clips(clips: list[dict]) -> list[dict]:
    """合并时间上重叠或相邻的片段。"""
    if not clips:
        return []

    sorted_clips = sorted(clips, key=lambda x: float(x["start"]))

    merged = [sorted_clips[0]]
    for clip in sorted_clips[1:]:
        last = merged[-1]
        if float(clip["start"]) <= float(last["end"]) + 5:
            last["end"] = max(float(last["end"]), float(clip["end"]))
            last["title"] = f"{last['title']} + {clip['title']}"
            last["reason"] = f"{last['reason']}; {clip['reason']}"
        else:
            merged.append(clip)

    return merged
