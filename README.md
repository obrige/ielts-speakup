# IELTS SpeakUP
基于 IELTS 官方评分标准的 DIY 口语评分系统。完全开源,跑在你自己的 8GB GPU 上,无需依赖任何云端 AI 服务(LLM 通过 API 调用,音频模型本地运行)。

## 🏗️ 架构概览

```
浏览器端(零 AI)
└─ MediaRecorder 录音 → POST 上传 WAV → 后端

后端:算法 + 45 次 LLM 调用
├─ 算法层(本地,无 LLM)
│  ├─ WhisperX        ASR + 词级对齐
│  ├─ librosa         F0/RMS/VAD/语速/停顿
│  ├─ wav2vec2        音素置信度 + 编辑距离
│  └─ TTR/MTLD/ngram  词汇/搭配/语法错误统计
│
├─ LLM 层(全部独立 session, JSON 输出)
│  ├─ 14 次 sub-dim   14 个子维度独立评分
│  ├─ 4 次 main-dim   4 个主维度独立综合
│  ├─ N 次 sentence   句子级反馈(每句独立)
│  ├─ M 次 ZPD        个性化范例(每个 issue 独立)
│  └─ 1 次 summary    总评生成
│
└─ 算法: Overall = (FC+LR+GRA+P)/4
```

## 📁 项目结构

```
quark/
├── docker-compose.yml          # WhisperX + API 服务编排
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── main.py                 # FastAPI 入口
│   └── scoring/
│       ├── whisperx_client.py  # WhisperX HTTP 客户端
│       ├── audio_metrics.py    # librosa + wav2vec2
│       ├── text_stats.py       # TTR/MTLD/grammar
│       ├── prompts.py          # 所有 LLM prompt + IELTS rubric
│       ├── llm_scorer.py       # 双模型并发调用
│       ├── pipeline.py         # 完整编排
│       └── questions.py        # 题库(可替换为外部服务)
└── frontend/
    ├── index.html              # 选题 + 录音
    ├── recorder.js             # MediaRecorder 上传
    └── report.html             # 评分报告
```

## 🚀 快速启动

### 0. 系统依赖

- Docker + Docker Compose
- NVIDIA GPU(8GB 显存)
- 16kHz 麦克风权限

### 1. 配置环境变量

**⚠️ 注意:docker-compose.yml 在项目根目录**

```bash
cd C:\quark
cd backend
cp .env.example .env
# 编辑 .env,填入 DeepSeek + Qwen API Key
```

### 2. 启动服务

```bash
# ⚠️ 回到项目根目录
cd C:\quark
docker-compose up -d --build
```

首次启动会下载 WhisperX 模型(约 1.5GB)和 wav2vec2 模型(约 360MB)。

### 3. 访问

打开浏览器:http://localhost:8080

## 🧪 验证

### 后端健康

```bash
curl http://localhost:8080/api/health
# {"status":"ok","pipeline_ready":true}
```

### 题库

```bash
curl "http://localhost:8080/api/questions?part=1"
```

### 评分(命令行)

```bash
curl -X POST http://localhost:8080/api/score \
  -F "audio=@test.wav" \
  -F "question_id=p1-hometown-1" \
  | jq .
```

## 💰 成本估算

| 项 | 单次 |
|---|---|
| 14 sub-dim LLM | ¥0.05 |
| 4 main-dim LLM | ¥0.03 |
| 15 句子反馈 LLM | ¥0.04 |
| 15 ZPD 范例 LLM | ¥0.05 |
| Summary LLM | ¥0.01 |
| **总计** | **¥0.18** |

按 1 万次/月计:**¥1,800/月**(LLM 部分;GPU 电费另计)

对比 BC 商业方案:本系统成本约 **1/100**,但评分精度同样受限(±0.5 band)。

## 🎓 评分维度

### 4 主维(对应 IELTS 官方)
- **FC** Fluency & Coherence(流利度与连贯性)
- **LR** Lexical Resource(词汇丰富度)
- **GRA** Grammatical Range & Accuracy(语法多样性与准确性)
- **P** Pronunciation(发音)

### 14 子维(本系统细化的诊断指标)
- **FC**: SpeechRate / PausingWordSearch / LogicalStructure / CohesiveDevices
- **LR**: LexicalRichness / Collocation / Paraphrasing / IdiomaticExpression
- **GRA**: GrammaticalStructure / GrammaticalErrors
- **P**: Intelligibility / Clarity / StressIntonationRhythm

## 🔬 与官方 IELTS 标准的对齐

`backend/scoring/prompts.py` 中包含完整官方 Band Descriptor 文本,所有 LLM 调用都基于此 rubric 判断。LLM 不使用任何编造的 band 阈值;每个分数都引用具体 descriptor 文本。

## 🛠️ 实施路线

- [x] **Step 1**: 核心算法 + 14+4 LLM 评分(本仓库)
- [ ] **Step 2**: 句级反馈高亮 + 浏览器交互
- [ ] **Step 3**: ZPD 个性化范例 + 练习历史
- [ ] **Step 4**: 数据收集 + Spearman 相关性验证 + calibration

## 📜 License

MIT