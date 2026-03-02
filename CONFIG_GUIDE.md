# CONFIG_GUIDE.md

> 本文档基于项目当前版本的 `config.txt`、`smart_anchor.py`、`api_server.py` 以及三个启动脚本：
> `start_lmdeploy.ps1 / start_api.ps1 / start_gptsovits.ps1` 编写，用于**可交付**的配置说明与落地指南。

---

## 1. 系统组成与数据流

本项目通常包含三个服务（对应三个启动脚本）：

1. **LMDeploy（LLM 推理服务）**  
   - 启动脚本：`start_lmdeploy.ps1`  
   - 提供 OpenAI 兼容接口：`/v1/chat/completions`、`/v1/models`  
   - `SmartVirtualAnchor` 会向 LMDeploy 发起对话请求（可流式）

2. **GPT-SoVITS（TTS 合成服务）**  
   - 启动脚本：`start_gptsovits.ps1`  
   - 提供 `/tts` 接口  
   - `SmartVirtualAnchor` 会把文本发送给 GPT-SoVITS 生成音频（wav 等）

3. **API Server（FastAPI + WebSocket + 静态资源）**  
   - 启动脚本：`start_api.ps1`  
   - 对外提供 WebSocket/HTTP 接口以及前端静态资源  
   - 内部会创建 `SmartVirtualAnchor`，并调用其 `chat_stream()` 进行对话与 TTS 分段推送

---

## 2. 配置读取与优先级（必须掌握）

### 2.1 `smart_anchor.py` 的优先级
**环境变量 > config.txt > 代码默认值**  
- 如果系统环境变量中存在 `KEY`，则优先使用环境变量
- 否则从 `config.txt` 读取
- 否则使用 `smart_anchor.py` 内的默认值

> 典型用途：在不改 `config.txt` 的情况下，临时用环境变量覆盖参数（例如 CI、不同机器、临时调试）。

### 2.2 `api_server.py` 的优先级
`api_server.py` 主要从**环境变量**读取参数；而 `start_api.ps1` 会把 `config.txt` 解析后导出为 `$env:*`，再启动 `python api_server.py`。  
因此，正常启动链路下你只需要维护一份 `config.txt`。

---

## 3. 参数分组详解（怎么设 / 有什么用 / 影响哪里）

> 说明：下文按功能分组，而不是强依赖 `config.txt` 的排列顺序。  
> 参数名均为环境变量/配置项的 key（例如 `LMDEPLOY_PORT=23333`）。

---

### 3.1 全局开关类

#### `RAG_ENABLED`（0/1）
- **作用**：是否启用长期记忆 / RAG（向量检索、持久化索引等）
- **怎么设**：
  - `0`：关闭（环境不具备 embedding 或不需要长期记忆时推荐）
  - `1`：开启（需要“可检索记忆”的场景）
- **影响模块**：`smart_anchor.py`（RAG memory 初始化与检索）

#### `EMOTION_ENABLED`（0/1）
- **作用**：是否启用“情绪标签”与情绪事件（与 Live2D 动作联动）
- **怎么设**：
  - `0`：纯文本/纯语音，不做情绪控制
  - `1`：要求 LLM 输出情绪标签（例如 `【EMO=开心】`）
- **影响模块**：`smart_anchor.py`、API Server 的流式解析与事件推送

#### `STREAM_LLM`（0/1）
- **作用**：LLM 是否采用流式输出
- **怎么设**：
  - `1`：流式（直播/交互推荐）
  - `0`：非流式（更容易调试，但互动延迟更大）
- **影响模块**：`smart_anchor.py` 的 `/v1/chat/completions` 调用方式

#### `STREAM_TTS_CHUNKED`（0/1）
- **作用**：是否开启“按标点分段送 TTS”（降低等待、提高直播感）
- **怎么设**：
  - `1`：开启（推荐）
  - `0`：整段一次性送 TTS（更稳定但更慢）

#### `ROLLING_SUMMARY_ENABLED`（0/1）
- **作用**：滚动摘要/压缩上下文的开关（用于长会话降 token）
- **建议**：当前如未接入完整摘要逻辑，可保持 `0`

#### `DISABLE_AUTO_OPEN`（0/1） + `AUTO_OPEN_DELAY_MS`
- **作用**：启动 API Server 后是否自动打开页面/延迟多久打开
- **怎么设**：
  - 服务器/无桌面环境：`DISABLE_AUTO_OPEN=1`
  - 本地开发：`DISABLE_AUTO_OPEN=0`，`AUTO_OPEN_DELAY_MS` 适当设置（如 600ms）

---

### 3.2 TTS（GPT-SoVITS）配置

#### 3.2.1 启动脚本相关（`start_gptsovits.ps1` 使用）
- `GPTSOVITS_DIR`：GPT-SoVITS 根目录（相对项目根）
- `GPTSOVITS_API`：启动入口脚本路径（相对项目根）
- `GPTSOVITS_HOST` / `GPTSOVITS_PORT`：监听地址/端口
- `GPTSOVITS_ARGS`：额外启动参数（以空格分隔）
- `GPTSOVITS_REDIRECT_LOG`：是否重定向日志到 `logs/` 文件
- `GPTSOVITS_URL`：TTS 服务 base URL（**SmartVirtualAnchor 实际调用这个**）

**常见设置**：
- 本机：`GPTSOVITS_URL=http://127.0.0.1:9880`
- 局域网：`GPTSOVITS_URL=http://<tts机器IP>:9880`（确保 API Server 可访问）

#### 3.2.2 合成参数（`/tts` payload）
- `TTS_REF_AUDIO_PATH`：参考音频路径（音色/风格基准）  
  **建议**：优先用绝对路径；确保 GPT-SoVITS 服务端能访问到该文件。
- `TTS_PROMPT_TEXT`：参考音频对应文本（用于对齐）
- `TTS_TEXT_LANG` / `TTS_PROMPT_LANG`：语言标记（如 `zh`）
- `TTS_MEDIA_TYPE`：返回媒体类型（推荐 `wav`）
- `TTS_STREAMING_MODE` / `TTS_STREAM_CHUNK_SIZE`：若需流式音频，开启并设置 chunk 大小

#### 3.2.3 分段、清洗、并发（强烈影响体验）
- `MIN_TTS_CHARS`：短句不发音（降低碎片音）
- `TTS_MAX_CHARS`：单段最大长度（避免超长合成卡顿）
- `MAX_PENDING_CHARS` / `FLUSH_MAX_SEGMENTS` / `POLL_INTERVAL_MS`：流式累积与触发分段的节流参数
- `TTS_WORKERS`：TTS 并发（建议 2~4，结合机器性能与 GPT-SoVITS 能力调整）
- `TTS_STRIP_MENTION`：去掉开头 `@xxx`
- `TTS_REMOVE_EMOJI`：移除 emoji 等不可读字符

---

### 3.3 LLM（LMDeploy）配置

#### 3.3.1 启动脚本相关（`start_lmdeploy.ps1` 使用）
- `LMDEPLOY_SUBCMD`：一般是 `serve api_server`
- `LMDEPLOY_MODEL_PATH`：本地模型路径  
  **重要**：脚本通常会检查路径存在，不存在会直接失败（避免误触在线下载）。
- `LMDEPLOY_BACKEND`：后端（如 `turbomind`）
- `LMDEPLOY_MODEL_FORMAT`：模型格式（必须与实际模型匹配）
- `LMDEPLOY_MODEL_NAME`：对外模型名（影响 `/v1/models` 与请求 `model` 字段）
- `LMDEPLOY_SERVER_NAME` / `LMDEPLOY_SERVER_PORT`：服务绑定地址/端口
- `LMDEPLOY_TP`：tensor parallel 并行度（单卡一般为 1）
- `LMDEPLOY_SESSION_LEN`：上下文长度（越大越吃显存）
- `LMDEPLOY_CACHE_MAX_ENTRY_COUNT`：缓存/显存相关参数（依 lmdeploy 版本行为而定）

#### 3.3.2 Anchor 调用相关
- `LMDEPLOY_URL`：显式指定 LLM base URL（推荐）  
  例：`http://127.0.0.1:23333`
- `LMDEPLOY_HOST` / `LMDEPLOY_PORT`：若没写 `LMDEPLOY_URL`，会用于拼接 URL

---

### 3.4 API Server（FastAPI）配置

#### 3.4.1 启动参数（`start_api.ps1` 使用）
- `API_ENTRY`：默认 `api_server.py`
- `API_HOST` / `API_PORT`：监听地址/端口
- `API_RELOAD`：热重载（开发 true、部署 false）
- `API_LOG_LEVEL`：日志级别（如 info/debug）

#### 3.4.2 Web 与 CORS
- `API_PUBLIC_BASE_URL`：对外可访问的 base url（前端拼接资源/链接时常用）
- `STATIC_WEB_PATH`：静态 web 目录（mount 到 `/web`）
- `STATIC_LIVE2D_PATH`：Live2D 静态资源目录（mount 到 `/Live2D`）

- `CORS_ALLOW_ORIGINS` / `CORS_ALLOW_CREDENTIALS` / `CORS_ALLOW_METHODS` / `CORS_ALLOW_HEADERS`  
  - 本地开发：可保持 `*`
  - 生产环境：建议收敛 allow_origins

#### 3.4.3 数据目录与默认会话
- `DATA_TTS_DIR`：TTS wav 输出目录（API Server 会创建/管理）
- `DATA_LIVE_SESSIONS_DIR`：直播会话/记忆落盘目录
- `DEFAULT_ROOM`：默认房间
- `DEFAULT_SENDER_NAME`：默认用户称呼

---

### 3.5 主播人格与采样参数（Prompt 相关）

#### 人格信息
- `ANCHOR_NAME`：主播名
- `ANCHOR_AGE`：主播年龄（提示信息）
- `ANCHOR_TRAITS`：特质（逗号分隔）
- `ANCHOR_INTERESTS`：兴趣（逗号分隔）
- `ANCHOR_STYLE`：说话风格提示
- `ANCHOR_SYSTEM_PROMPT`：强约束 system prompt
- `ANCHOR_PERSONALITY`：人格补充
- `ANCHOR_WELCOME_MESSAGE` / `ANCHOR_GOODBYE_MESSAGE`：欢迎/告别

#### LLM 采样参数
- `ANCHOR_TEMPERATURE`：随机性（0.6~0.9 常用）
- `ANCHOR_TOP_P`：采样截断（0.8~0.95 常用）
- `ANCHOR_MAX_TOKENS`：模型生成 token 上限
- `ANCHOR_MAX_RESPONSE_LENGTH`：最终字符级截断上限（避免超长回复）

---

### 3.6 知识库 / Grounded / RAG

#### TXT 知识库
- `KNOWLEDGE_TXT`：`knowledge.txt` 路径
- `KB_TOP_K`：检索 top_k
- `KB_SCORE_THRESHOLD_PROMPT`：进入 prompt 的阈值（更严格）
- `KB_SCORE_THRESHOLD_CHAT`：判断是否命中的阈值（更宽松）
- `KB_FORCE_GROUNDED_ON_HITS`：命中时强制 grounded（减少编造）

#### 长期记忆 / RAG
- `RAG_ALLOW_DOWNLOAD`：允许自动下载 embedding（离线建议 0）
- `RAG_EMBED_MODEL`：embedding 模型（可为本地路径）
- `RAG_PERSIST_ROOT`：索引持久化根目录
- `RAG_SIMILARITY_TOP_K` / `RAG_SEARCH_TOP_K`：检索 top_k

---

### 3.7 情绪标签与 Live2D 映射

- `EMOTION_LABELS`：支持的情绪标签集合（逗号分隔）
- `EMOTION_DEFAULT`：默认情绪（标签缺失时兜底）
- `EMOTION_SYSTEM_PROMPT`：要求输出情绪标签的提示模板
- `EMOTION_TAG_REQUIRED`：是否要求必须输出情绪标签
- `EMOTION_MAP_<情绪名>=动作,参数`：情绪到 Live2D 动作映射  
  例：`EMOTION_MAP_开心=Tap,1|FlickUp,0`

---

### 3.8 会话与记忆（短期上下文）

- `MEMORY_KEEP_LAST`：内存保留最近 N 条消息（超过裁剪）
- `PROMPT_HISTORY_MAX_MESSAGES`：拼 prompt 时最多带的历史条数（影响模型“记忆感”）

---

### 3.9 超时与缓存（稳定性关键）

- `HTTP_TIMEOUT_MODELS`：拉取 `/v1/models` 超时
- `HTTP_TIMEOUT_CHAT`：非流式对话超时
- `HTTP_TIMEOUT_STREAM`：流式对话超时（一般要更大）
- `HTTP_TIMEOUT_TTS`：TTS 请求超时
- `MODEL_NAME_CACHE_TTL`：模型名缓存 TTL（减少频繁请求 `/v1/models`）

---

### 3.10 调试参数（排障用）

- `DEBUG_SHOW_LLM_INPUT`：打印送入 LLM 的 prompt
- `DEBUG_PROMPT_MAX_CHARS`：打印 prompt 的最大长度
- `DEBUG_TTS` / `DEBUG_EMOTION` / `DEBUG_LLM_STREAM`：分模块调试开关
- `DEBUG_LLM_MAX_CHARS`：打印 LLM 输出片段的最大长度

---

## 4. 三个启动脚本的字段对照

### 4.1 `start_lmdeploy.ps1`
典型读取并使用：
- `LMDEPLOY_SUBCMD`
- `LMDEPLOY_MODEL_PATH`
- `LMDEPLOY_BACKEND`
- `LMDEPLOY_MODEL_FORMAT`
- `LMDEPLOY_MODEL_NAME`
- `LMDEPLOY_SERVER_NAME` / `LMDEPLOY_SERVER_PORT`
- `LMDEPLOY_TP`
- `LMDEPLOY_SESSION_LEN`
- `LMDEPLOY_CACHE_MAX_ENTRY_COUNT`

### 4.2 `start_gptsovits.ps1`
典型读取并使用：
- `GPTSOVITS_DIR`
- `GPTSOVITS_API`
- `GPTSOVITS_HOST` / `GPTSOVITS_PORT`
- `GPTSOVITS_ARGS`
- `GPTSOVITS_REDIRECT_LOG`
- `GPTSOVITS_URL`
- `TTS_REF_AUDIO_PATH`（常做路径解析/校验）

### 4.3 `start_api.ps1`
典型读取并使用：
- `API_ENTRY`
- `API_HOST` / `API_PORT`
- `API_RELOAD`
- `API_LOG_LEVEL`

---

## 5. 推荐配置组合（直接改值可用）

### 5.1 本机一体化（最推荐）
- `LMDEPLOY_URL=http://127.0.0.1:23333`
- `GPTSOVITS_URL=http://127.0.0.1:9880`
- `API_HOST=0.0.0.0` `API_PORT=8000`

启动顺序：
1. `start_lmdeploy.ps1`
2. `start_gptsovits.ps1`
3. `start_api.ps1`

### 5.2 LMDeploy 独立一台机器
- API Server 机器：`LMDEPLOY_URL=http://<lmdeploy机器IP>:23333`
- LMDeploy 机器：`LMDEPLOY_SERVER_NAME=0.0.0.0`

### 5.3 只验证文字，不启 TTS
- `EMOTION_ENABLED=0`（可选）
- 前端请求时 `tts=false`（或不启动 `start_gptsovits.ps1`）

---

## 6. 排障清单（最常见问题）

1. **LMDeploy 启不起来**  
   - 检查 `LMDEPLOY_MODEL_PATH` 是否存在、是否为正确目录
   - 检查 `LMDEPLOY_MODEL_FORMAT` 是否与模型匹配

2. **API Server 连不上 LLM**  
   - 确认 `LMDEPLOY_URL` 可从 API Server 机器访问（浏览器/`curl`）
   - 确认端口/防火墙/绑定地址正确（`LMDEPLOY_SERVER_NAME=0.0.0.0`）

3. **TTS 失败或无声音**  
   - 检查 `GPTSOVITS_URL` 是否正确
   - 检查 `TTS_REF_AUDIO_PATH` 是否存在且服务端可访问
   - 适当增大 `HTTP_TIMEOUT_TTS`

4. **直播感不强（等很久才出声音）**  
   - 开启：`STREAM_LLM=1`、`STREAM_TTS_CHUNKED=1`
   - 调小 `MIN_TTS_CHARS`、调大 `FLUSH_MAX_SEGMENTS`（在可接受范围内）

---

## 7. 维护建议

- 生产环境建议把 `config.txt` 纳入版本管理，并为不同部署环境准备：
  - `config.dev.txt`
  - `config.prod.txt`
  - `config.gpuA.txt` 等
- 通过环境变量覆盖少量差异（如 URL、端口、路径），减少维护成本。

---

**END**
