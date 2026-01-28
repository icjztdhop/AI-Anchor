# 项目配置说明（README.md）

本文档用于说明项目中各类 **环境变量 / 配置参数** 的含义与用途，便于部署、维护与二次开发。  
配置文件通常位于 `start_all.bat` 所在目录。

---

## 一、全局配置

| 参数名 | 说明 | 取值示例 |
|------|------|---------|
| `RAG_ENABLED` | 是否启用 RAG（检索增强生成）。启用后会引入 `torch`，启动速度较慢 | `0` 关闭 / `1` 启用 |

---

## 二、GPT-SoVITS（TTS 语音合成服务）

### 1. 服务与路径配置

| 参数名 | 说明 | 示例 |
|------|------|------|
| `GPTSOVITS_DIR` | GPT-SoVITS 项目目录 | `GPT-SoVITS` |
| `GPTSOVITS_API` | TTS API 启动脚本路径 | `GPT-SoVITS\api_v2.py` |
| `GPTSOVITS_HOST` | TTS 服务监听地址 | `0.0.0.0` |
| `GPTSOVITS_PORT` | TTS 服务端口 | `9880` |
| `GPTSOVITS_REDIRECT_LOG` | 是否重定向日志输出 | `0 / 1` |
| `GPTSOVITS_ARGS` | 启动附加参数 | （可空） |

### 2. TTS 调用参数

| 参数名 | 说明 | 示例 |
|------|------|------|
| `GPTSOVITS_URL` | TTS 服务访问地址 | `http://127.0.0.1:9880` |
| `TTS_REF_AUDIO_PATH` | 参考音频路径（用于音色克隆） | `GPT-SoVITS\ref\default.wav` |
| `TTS_PROMPT_TEXT` | 参考音频对应文本 | `为什么抛弃更高效更坚固的形态？` |

---

## 三、LMDeploy（大语言模型服务）

### 1. 模型与后端

| 参数名 | 说明 | 示例 |
|------|------|------|
| `LMDEPLOY_SUBCMD` | LMDeploy 启动命令 | `serve api_server` |
| `LMDEPLOY_MODEL_PATH` | 模型路径 | `model\internlm2_5-7b-chat-int4` |
| `LMDEPLOY_BACKEND` | 推理后端 | `turbomind` |
| `LMDEPLOY_MODEL_FORMAT` | 模型格式 | `awq` |
| `LMDEPLOY_MODEL_NAME` | 模型名称标识 | `internlm2_5_7b_chat_awq_int4` |

### 2. 服务与性能参数

| 参数名 | 说明 | 示例 |
|------|------|------|
| `LMDEPLOY_SERVER_NAME` | 服务监听地址 | `0.0.0.0` |
| `LMDEPLOY_SERVER_PORT` | 服务端口 | `23333` |
| `LMDEPLOY_TP` | Tensor Parallel 数量 | `1` |
| `LMDEPLOY_SESSION_LEN` | 最大上下文长度 | `2048` |
| `LMDEPLOY_CACHE_MAX_ENTRY_COUNT` | KV Cache 占用比例 | `0.35` |

### 3. API 与健康检查

| 参数名 | 说明 | 示例 |
|------|------|------|
| `LMDEPLOY_HOST` | API 访问地址 | `127.0.0.1` |
| `LMDEPLOY_PORT` | API 访问端口 | `23333` |

---

## 四、API Server（对外接口）

| 参数名 | 说明 | 示例 |
|------|------|------|
| `API_ENTRY` | API 入口文件 | `api_server.py` |
| `API_HOST` | API 监听地址 | `0.0.0.0` |
| `API_PORT` | API 服务端口 | `8000` |
| `API_RELOAD` | 是否启用热重载 | `false` |
| `API_LOG_LEVEL` | 日志级别 | `info` |

### 浏览器自动打开

| 参数名 | 说明 | 示例 |
|------|------|------|
| `DISABLE_AUTO_OPEN` | 是否禁用自动打开浏览器 | `0 / 1` |
| `AUTO_OPEN_DELAY_MS` | 自动打开延迟（毫秒） | `600` |

---

## 五、CORS 与静态资源

### CORS 设置

| 参数名 | 说明 | 示例 |
|------|------|------|
| `CORS_ALLOW_ORIGINS` | 允许的来源 | `*` |
| `CORS_ALLOW_CREDENTIALS` | 是否允许携带凭证 | `true` |
| `CORS_ALLOW_METHODS` | 允许的 HTTP 方法 | `*` |
| `CORS_ALLOW_HEADERS` | 允许的请求头 | `*` |

### 静态资源路径

| 参数名 | 说明 | 示例 |
|------|------|------|
| `STATIC_WEB_PATH` | Web 静态资源目录 | `web` |
| `STATIC_LIVE2D_PATH` | Live2D 资源目录 | `Live2D` |

---

## 六、数据目录与默认会话

| 参数名 | 说明 | 示例 |
|------|------|------|
| `DATA_TTS_DIR` | TTS 数据输出目录 | `data/tts` |
| `DATA_LIVE_SESSIONS_DIR` | 直播会话数据目录 | `data/live_sessions` |
| `DEFAULT_ROOM` | 默认房间名 | `default_room` |
| `DEFAULT_SENDER_NAME` | 默认用户名称 | `路人甲` |

---

## 七、主播人格 / 知识库（Smart Anchor）

| 参数名 | 说明 | 示例 |
|------|------|------|
| `ANCHOR_NAME` | 主播名称 | 小爱 |
| `ANCHOR_AGE` | 主播年龄 | 22 |
| `ANCHOR_TRAITS` | 性格特征 | 活泼,开朗,细心 |
| `ANCHOR_INTERESTS` | 兴趣爱好 | 游戏,音乐,动漫 |
| `ANCHOR_STYLE` | 说话风格 | 热情、互动强、有梗 |
| `ANCHOR_SYSTEM_PROMPT` | 系统提示词 | 始终以主播身份回应 |
| `ANCHOR_PERSONALITY` | 人格描述 | 亲切自然，不胡编事实 |
| `ANCHOR_WELCOME_MESSAGE` | 欢迎语 | 欢迎来到我的直播间！ |
| `ANCHOR_GOODBYE_MESSAGE` | 结束语 | 今天的直播就到这里啦 |
| `ANCHOR_MAX_RESPONSE_LENGTH` | 最大回复长度 | 500 |
| `ANCHOR_TEMPERATURE` | 回复随机性 | 0.7 |
| `ANCHOR_TOP_P` | 核采样参数 | 0.9 |
| `KNOWLEDGE_TXT` | 知识库文件 | `knowledge.txt` |

---

## 八、流式输出与 TTS 行为

| 参数名 | 说明 | 示例 |
|------|------|------|
| `STREAM_LLM` | 是否启用 LLM 流式输出 | `1` |
| `STREAM_TTS_CHUNKED` | TTS 是否分段输出 | `1` |
| `TTS_WORKERS` | TTS 并发线程数 | `2` |
| `MIN_TTS_CHARS` | 触发 TTS 的最小字符数 | `6` |
| `MAX_PENDING_CHARS` | 最大待处理字符数 | `160` |
| `FLUSH_MAX_SEGMENTS` | 最大刷新分段数 | `2` |
| `POLL_INTERVAL_MS` | 轮询间隔（毫秒） | `10` |

---

## 九、情绪系统（Emotion → TTS / Live2D）

### 基础配置

| 参数名 | 说明 | 示例 |
|------|------|------|
| `EMOTION_ENABLED` | 是否启用情绪系统 | `1` |
| `EMOTION_LABELS` | 支持的情绪标签 | 正常,开心,失落,疑惑,惊讶,生气 |
| `EMOTION_DEFAULT` | 默认情绪 | 正常 |
| `EMOTION_SYSTEM_PROMPT` | 情绪输出规则 | 回复末尾必须包含情绪标签 |

### Live2D 动作映射

| 参数名 | 说明 | 示例 |
|------|------|------|
| `EMOTION_MAP_正常` | 正常情绪动作 | `Tap,0` |
| `EMOTION_MAP_开心` | 开心情绪动作 | `Tap,1\|FlickUp,0` |
| `EMOTION_MAP_失落` | 失落情绪动作 | `Flick@Body,0` |
| `EMOTION_MAP_疑惑` | 疑惑情绪动作 | `Flick,0` |
| `EMOTION_MAP_惊讶` | 惊讶情绪动作 | `FlickDown,0` |
| `EMOTION_MAP_生气` | 生气情绪动作 | `Tap@Body,0` |

### 调试参数

| 参数名 | 说明 | 示例 |
|------|------|------|
| `DEBUG_TTS` | TTS 调试日志 | `1` |
| `DEBUG_EMOTION` | 情绪调试日志 | `1` |

---

## 十、健康检查

| 参数名 | 说明 | 示例 |
|------|------|------|
| `HEALTH_CHECK_TIMEOUT_SECONDS` | 健康检查超时时间（秒） | `5` |

---

> 建议：  
> - **生产环境**关闭 `DEBUG_*` 相关参数  
> - **低显存设备**合理降低 `SESSION_LEN` 与并发数  
> - **直播场景**建议开启流式输出与情绪系统

