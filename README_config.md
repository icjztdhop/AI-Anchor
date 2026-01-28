# 项目配置说明（README_config.md）

本文档**严格依据当前 `config.txt` 内容**整理生成，用于说明各配置参数的含义、用途与推荐用法，便于部署、维护与二次开发。

> 配置优先级：  
> **环境变量（env） ＞ config.txt ＞ 代码默认值**

---

## 一、全局配置

```ini
RAG_ENABLED=1
```

| 参数名 | 说明 | 示例 |
|---|---|---|
| RAG_ENABLED | 是否启用 RAG（检索增强生成）。启用后会引入 torch，启动较慢 | 0 / 1 |

---

## 二、GPT-SoVITS（TTS 服务）

### 1. 服务启动相关（start_all.bat 使用）

```ini
GPTSOVITS_DIR=GPT-SoVITS
GPTSOVITS_API=GPT-SoVITS\api_v2.py
GPTSOVITS_HOST=0.0.0.0
GPTSOVITS_PORT=9880
GPTSOVITS_REDIRECT_LOG=0
GPTSOVITS_ARGS=
```

| 参数名 | 说明 |
|---|---|
| GPTSOVITS_DIR | GPT-SoVITS 项目目录 |
| GPTSOVITS_API | API 启动脚本路径 |
| GPTSOVITS_HOST | 服务监听地址 |
| GPTSOVITS_PORT | 服务端口 |
| GPTSOVITS_REDIRECT_LOG | 是否重定向日志 |
| GPTSOVITS_ARGS | 启动附加参数 |

---

### 2. smart_anchor 调用参数

```ini
GPTSOVITS_URL=http://127.0.0.1:9880
TTS_REF_AUDIO_PATH=GPT-SoVITS\ref\default.wav
TTS_PROMPT_TEXT=为什么抛弃更高效更坚固的形态？
```

| 参数名 | 说明 |
|---|---|
| GPTSOVITS_URL | TTS 服务访问地址 |
| TTS_REF_AUDIO_PATH | 参考音频路径（音色克隆） |
| TTS_PROMPT_TEXT | 参考音频对应文本 |

---

### 3. TTS 生成参数（smart_anchor.py 读取）

```ini
TTS_TEXT_LANG=zh
TTS_PROMPT_LANG=zh
TTS_MEDIA_TYPE=wav
TTS_STREAMING_MODE=0
TTS_STREAM_CHUNK_SIZE=65536
```

| 参数名 | 说明 | 示例 |
|---|---|---|
| TTS_TEXT_LANG | 合成文本语言 | zh |
| TTS_PROMPT_LANG | Prompt 文本语言 | zh |
| TTS_MEDIA_TYPE | 输出音频格式 | wav |
| TTS_STREAMING_MODE | 一次性 TTS 是否使用 streaming | 0 |
| TTS_STREAM_CHUNK_SIZE | 流式音频 chunk 大小（bytes） | 65536 |

---

## 三、LMDeploy（大语言模型服务）

### 1. 模型与后端

```ini
LMDEPLOY_SUBCMD=serve api_server
LMDEPLOY_MODEL_PATH=model\internlm2_5-7b-chat-int4
LMDEPLOY_BACKEND=turbomind
LMDEPLOY_MODEL_FORMAT=awq
LMDEPLOY_MODEL_NAME=internlm2_5_7b_chat_awq_int4
```

---

### 2. 服务与性能参数

```ini
LMDEPLOY_SERVER_NAME=0.0.0.0
LMDEPLOY_SERVER_PORT=23333
LMDEPLOY_TP=1
LMDEPLOY_SESSION_LEN=2048
LMDEPLOY_CACHE_MAX_ENTRY_COUNT=0.35
```

---

### 3. API 访问

```ini
LMDEPLOY_HOST=127.0.0.1
LMDEPLOY_PORT=23333
```

---

## 四、API Server（对外接口）

```ini
API_ENTRY=api_server.py
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
API_LOG_LEVEL=info
```

---

### 浏览器自动打开

```ini
DISABLE_AUTO_OPEN=0
AUTO_OPEN_DELAY_MS=600
```

---

## 五、CORS 与静态资源

```ini
CORS_ALLOW_ORIGINS=*
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=*
CORS_ALLOW_HEADERS=*

STATIC_WEB_PATH=web
STATIC_LIVE2D_PATH=Live2D
```

---

## 六、数据目录与默认会话

```ini
DATA_TTS_DIR=data/tts
DATA_LIVE_SESSIONS_DIR=data/live_sessions
DEFAULT_ROOM=default_room
DEFAULT_SENDER_NAME=路人甲
```

---

## 七、主播人格 / 知识库（smart_anchor）

```ini
ANCHOR_NAME=小爱
ANCHOR_AGE=22
ANCHOR_TRAITS=活泼,开朗,细心
ANCHOR_INTERESTS=游戏,音乐,动漫
ANCHOR_STYLE=直播间口吻：热情、会@观众、互动感强，简短有梗
ANCHOR_SYSTEM_PROMPT=你正在直播，请始终以主播身份回应观众，不要提及你是AI模型。
ANCHOR_PERSONALITY=你是一个亲切自然的虚拟主播，说话口语化，喜欢和观众互动，但不会胡编事实。
ANCHOR_WELCOME_MESSAGE=欢迎来到我的直播间！有什么想聊的吗？
ANCHOR_GOODBYE_MESSAGE=今天的直播就到这里啦，大家晚安！
```

---

### LLM 生成参数

```ini
ANCHOR_MAX_RESPONSE_LENGTH=500
ANCHOR_TEMPERATURE=0.7
ANCHOR_TOP_P=0.9
ANCHOR_MAX_TOKENS=300
```

---

### 知识库

```ini
KNOWLEDGE_TXT=knowledge.txt
```

---

## 八、流式 / TTS 行为控制

```ini
STREAM_LLM=1
STREAM_TTS_CHUNKED=1

TTS_WORKERS=2
MIN_TTS_CHARS=6
MAX_PENDING_CHARS=160
FLUSH_MAX_SEGMENTS=2
POLL_INTERVAL_MS=10
```

---

## 九、情绪系统（Emotion → TTS / Live2D）

### 基础配置

```ini
EMOTION_ENABLED=1
EMOTION_LABELS=正常,开心,失落,疑惑,惊讶,生气
EMOTION_DEFAULT=正常
```

### 情绪系统提示词

```ini
EMOTION_SYSTEM_PROMPT=你是一个有情感的主播，请在每次回复的末尾添加一个情感标签...
```

### Live2D 动作映射

```ini
EMOTION_MAP_正常=Tap,0
EMOTION_MAP_开心=Tap,1|FlickUp,0
EMOTION_MAP_失落=Flick@Body,0
EMOTION_MAP_疑惑=Flick,0
EMOTION_MAP_惊讶=FlickDown,0
EMOTION_MAP_生气=Tap@Body,0
```

---

## 十、调试参数

```ini
DEBUG_TTS=1
DEBUG_EMOTION=1
```

| 参数名 | 说明 | 示例 |
|---|---|---|
| DEBUG_TTS | TTS 调试日志 | 1 |
| DEBUG_EMOTION | 情绪调试日志 | 1 |

---

## 十一、健康检查

```ini
HEALTH_CHECK_TIMEOUT_SECONDS=5
```

| 参数名 | 说明 | 示例 |
|---|---|---|
| HEALTH_CHECK_TIMEOUT_SECONDS | 健康检查超时时间（秒） | 5 |

---

## 十二、使用建议

- 生产环境建议关闭 `DEBUG_*` 相关参数  
- 显存较小设备建议降低 `LMDEPLOY_SESSION_LEN` 与并发数  
- 直播场景建议开启流式输出与情绪系统  

---

> 本 README 与 `config.txt` 保持一一对应，如新增配置项，请同步更新本文档。
