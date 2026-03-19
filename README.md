# VideoCuter

一个基于 AI 的教育类直播视频智能切片工具。自动从直播回放中识别有价值的教育内容片段（升学讨论、知识分享、精彩互动等），过滤掉营销话术和无关内容，并切割为独立的短视频。

## 工作流程

```
视频文件 → 提取音频 → 语音转文字(带时间戳) → AI分析有价值片段 → 视频切片 → 输出
```

1. **提取音频** — 使用 FFmpeg 从视频中提取 16kHz 单声道 WAV 音频
2. **语音转文字** — 使用 faster-whisper 本地模型将音频转录为带时间戳的文本，支持 CUDA 加速
3. **AI 分析** — 将转录文本发送给大语言模型，识别有价值的片段并返回 JSON
4. **视频切片** — 使用 FFmpeg 按时间段进行无损切片，输出多个短视频

## 环境要求

- Python >= 3.14
- [FFmpeg](https://ffmpeg.org/)（需要预先安装并加入 PATH）
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- （可选）NVIDIA 显卡 + CUDA 12.x + cuDNN 8.x，用于加速语音转文字

## 安装

```bash
git clone <repo-url>
cd VideoCuter
uv sync
```

## 配置

复制配置示例文件并修改：

```bash
cp config.example.json config.json
```

### 云端 API（如 DeepSeek）

```json
{
  "api_key": "your-api-key-here",
  "base_url": "https://api.deepseek.com/v1",
  "model": "deepseek-chat",
  "use_json_format": true,
  "whisper_model": "large-v3",
  "language": "zh"
}
```

### 本地 Ollama

先确保 [Ollama](https://ollama.com/) 已安装并运行，然后拉取模型：

```bash
ollama pull qwen2.5:14b
ollama serve
```

配置文件：

```json
{
  "api_key": "",
  "base_url": "http://localhost:11434/v1",
  "model": "qwen2.5:14b",
  "use_json_format": false,
  "whisper_model": "large-v3",
  "language": "zh"
}
```

> `use_json_format` 设为 `false` 是因为部分 Ollama 模型不支持 JSON 模式，程序会自动从回复中提取 JSON，并在解析失败时自动重试一次。

### 配置项说明

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `api_key` | API 密钥，使用 Ollama 时留空 | - |
| `base_url` | API 地址 | - |
| `model` | 大语言模型名称 | `deepseek-chat` |
| `use_json_format` | 是否使用 JSON 模式（Ollama 建议关闭） | `true` |
| `whisper_model` | Whisper 模型大小（`tiny`/`base`/`small`/`medium`/`large-v3`） | `large-v3` |
| `language` | 语音识别语言 | `zh` |

## 使用

```bash
# 基本用法
uv run python main.py /path/to/video.mp4

# 指定输出目录
uv run python main.py /path/to/video.mp4 /path/to/output
```

输出文件将保存在视频同目录的 `clips/` 文件夹下（或指定的输出目录）。

## AI 分析规则

### 保留的有价值片段

- 升学相关的话题讨论（择校、志愿填报、升学政策解读、考试备考经验等）
- 有价值的知识分享或教学内容（学科知识讲解、学习方法、规划建议等）
- 精彩的互动或对话（主播与观众之间有实质内容的问答互动）
- 有争议或引发讨论的话题（教育观点碰撞、不同升学路径的利弊分析等）

### 自动过滤的内容

- 无意义的寒暄和重复的问候
- 产品展示及相关营销话术
- 转发领取资料的营销话术
- 机构课程介绍及招生宣传
- 直播赠送物料展示和抽奖环节
- 引导关注、点赞、分享等互动指令

## 输出文件

运行后会生成以下文件：

```
├── video.wav                # 提取的音频文件
├── video.json               # 转录结果（带时间戳的文本，可复用跳过重复转录）
├── video.analysis.json      # AI 分析结果（有价值片段列表，可复用跳过重复分析）
└── clips/
    ├── 001_片段标题.mp4      # 切片视频 1
    ├── 002_片段标题.mp4      # 切片视频 2
    └── ...
```

> 再次运行同一视频时，若 `video.json` 或 `video.analysis.json` 已存在，程序会自动跳过对应步骤，直接使用缓存结果。

## 项目结构

```
VideoCuter/
├── main.py                        # 主程序入口
├── config.json                    # 配置文件（不纳入版本控制）
├── config.example.json            # 配置示例
├── videocuter/
│   ├── audio_extractor.py         # 音频提取
│   ├── transcriber.py             # 语音转文字（子进程执行，支持 CUDA）
│   ├── analyzer.py                # AI 内容分析
│   └── video_cutter.py            # 视频切片
├── pyproject.toml
└── uv.lock
```

## 注意事项

- 首次运行时 faster-whisper 会自动下载 Whisper 模型（large-v3 约 3GB），请确保网络通畅
- `large-v3` 模型在 CPU 上运行较慢，有 NVIDIA 显卡时会自动切换到 CUDA 加速；如无 GPU 且速度要求不高，可将 `whisper_model` 改为 `base` 或 `small`
- CUDA 加速要求 CUDA >= 12.x，ctranslate2 目前暂不支持 CUDA 13
- 长视频的转录文本可能较大，程序会自动分段发送给大模型分析
- 无损切片速度极快，但切点可能在关键帧处有几秒偏差
- Windows 用户无需额外配置，FFmpeg 和 Python 依赖均支持 Windows
