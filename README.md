<div align="center">
<img alt="icon-512.png" height="80" src="web/favicon.png"/>
<h1>AI Anchor</h1></div>


# AI Anchor
**AI Anchor虚拟主播，使用LMDeploy作为LLM服务端，使用GPT-SoVITS作为TTS服务端，使用LlamaIndex作为记忆管理器，使用网页版Live2d展示虚拟主播形象，完全本地部署，音频流式输出，最快1s返回首段语音。**
> 在NVIDIA GeForce RTX 4070ti 12G显卡上，使用internlm2_5-7b-chat-int4模型(原模型int4量化)，可以实现1s返回首段语音
> 目前采用完整的GPT-SoVITS包实现TTS服务，请单独下载[GPT-SoVITS的离线包](https://github.com/RVC-Boss/GPT-SoVITS/releases)，后续会单独抽离推理部分。
> **代码大部分由AI编写，有BUG很正常**
## 使用方法
 1. 克隆仓库文件  `git clone https://github.com/icjztdhop/AI-Anchor.git` 
 2. 切换工作目录  `cd`  或者直接在代码文件夹右键打开cmd。
 3. 创建虚拟环境  `python -m venv venv` ，若没有安装python可以使用python便携包，使用  `"C:python——path\python.exe" -m venv venv`  一样能创建虚拟环境。
 4. 进入虚拟环境  `.\venv\Scripts\Activate.ps1`  ，成功进入虚拟环境则会出现绿色  `(venv)`  字样在命令行最前。
 5. 安装依赖  `pip install -r requirements.txt`  或清华源  `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`  
 6. 新建  `model`  文件夹，放入[LMDeploy支持的模型](https://lmdeploy.readthedocs.io/zh-cn/v0.5.2/supported_models/supported_models.html)，推荐使用[LMDeploy进行量化]。(https://lmdeploy.readthedocs.io/zh-cn/v0.5.2/quantization/w4a16.html)后部署，让消费级显卡跑上更大的模型
 7. 下载[GPT-SoVITS的离线包](https://github.com/RVC-Boss/GPT-SoVITS/releases)，文件夹命名为  `GPT-SoVITS`  整个放入即可。
 8. 创建  `Live2D`  文件夹，结构为Live2D/模型名称/模型名称.model3.json
 9. （可选）向  `knowledge.txt`  注入知识库
 10.  编辑  `config.txt`  中的启动参数，若不懂参照后面的参数解释，仅编辑模型名称也能够正常启动。
 11. windows：双击  `run.bat`  或者直接把  `run.ps1`  拖入命令行按回车，linux：命令行输入  `./run.sh`
 12. 会新启动3个命令行，每个命令行都无报错就运行成功。
 13. 会自动打开UI界面，在右侧加载模型处填入类似  `/Live2D/hiyori/hiyori_pro_t11.model3.json`  的模型路径，点击加载模型按钮即可加载模型（目前测试公模  `hiyori_pro_t11`  表现正常）。
 14. 点击左侧  `连接ws`  按钮即可开启对话。
## config参数
### 一、全局配置

| 参数名 | 说明 | 取值示例 |
|------|------|---------|
| `RAG_ENABLED` | 是否启用 RAG（检索增强生成）。启用后会引入 `torch`，启动速度较慢 | `0` 关闭 / `1` 启用 |

---

### 二、GPT-SoVITS（TTS 语音合成服务）

#### 1. 服务与路径配置

| 参数名 | 说明 | 示例 |
|------|------|------|
| `GPTSOVITS_DIR` | GPT-SoVITS 项目目录 | `GPT-SoVITS` |
| `GPTSOVITS_API` | TTS API 启动脚本路径 | `GPT-SoVITS\api_v2.py` |
| `GPTSOVITS_HOST` | TTS 服务监听地址 | `0.0.0.0` |
| `GPTSOVITS_PORT` | TTS 服务端口 | `9880` |
| `GPTSOVITS_REDIRECT_LOG` | 是否重定向日志输出 | `0 / 1` |
| `GPTSOVITS_ARGS` | 启动附加参数 | （可空） |

#### 2. TTS 调用参数

| 参数名 | 说明 | 示例 |
|------|------|------|
| `GPTSOVITS_URL` | TTS 服务访问地址 | `http://127.0.0.1:9880` |
| `TTS_REF_AUDIO_PATH` | 参考音频路径（用于音色克隆） | `GPT-SoVITS\ref\default.wav` |
| `TTS_PROMPT_TEXT` | 参考音频对应文本 | `为什么抛弃更高效更坚固的形态？` |

---

### 三、LMDeploy（大语言模型服务）

#### 1. 模型与后端

| 参数名 | 说明 | 示例 |
|------|------|------|
| `LMDEPLOY_SUBCMD` | LMDeploy 启动命令 | `serve api_server` |
| `LMDEPLOY_MODEL_PATH` | 模型路径 | `model\internlm2_5-7b-chat-int4` |
| `LMDEPLOY_BACKEND` | 推理后端 | `turbomind` |
| `LMDEPLOY_MODEL_FORMAT` | 模型格式 | `awq` |
| `LMDEPLOY_MODEL_NAME` | 模型名称标识 | `internlm2_5_7b_chat_awq_int4` |

#### 2. 服务与性能参数

| 参数名 | 说明 | 示例 |
|------|------|------|
| `LMDEPLOY_SERVER_NAME` | 服务监听地址 | `0.0.0.0` |
| `LMDEPLOY_SERVER_PORT` | 服务端口 | `23333` |
| `LMDEPLOY_TP` | Tensor Parallel 数量 | `1` |
| `LMDEPLOY_SESSION_LEN` | 最大上下文长度 | `2048` |
| `LMDEPLOY_CACHE_MAX_ENTRY_COUNT` | KV Cache 占用比例 | `0.35` |

#### 3. API 与健康检查

| 参数名 | 说明 | 示例 |
|------|------|------|
| `LMDEPLOY_HOST` | API 访问地址 | `127.0.0.1` |
| `LMDEPLOY_PORT` | API 访问端口 | `23333` |

---

### 四、API Server（对外接口）

| 参数名 | 说明 | 示例 |
|------|------|------|
| `API_ENTRY` | API 入口文件 | `api_server.py` |
| `API_HOST` | API 监听地址 | `0.0.0.0` |
| `API_PORT` | API 服务端口 | `8000` |
| `API_RELOAD` | 是否启用热重载 | `false` |
| `API_LOG_LEVEL` | 日志级别 | `info` |

#### 浏览器自动打开

| 参数名 | 说明 | 示例 |
|------|------|------|
| `DISABLE_AUTO_OPEN` | 是否禁用自动打开浏览器 | `0 / 1` |
| `AUTO_OPEN_DELAY_MS` | 自动打开延迟（毫秒） | `600` |

---

### 五、CORS 与静态资源

#### CORS 设置

| 参数名 | 说明 | 示例 |
|------|------|------|
| `CORS_ALLOW_ORIGINS` | 允许的来源 | `*` |
| `CORS_ALLOW_CREDENTIALS` | 是否允许携带凭证 | `true` |
| `CORS_ALLOW_METHODS` | 允许的 HTTP 方法 | `*` |
| `CORS_ALLOW_HEADERS` | 允许的请求头 | `*` |

#### 静态资源路径

| 参数名 | 说明 | 示例 |
|------|------|------|
| `STATIC_WEB_PATH` | Web 静态资源目录 | `web` |
| `STATIC_LIVE2D_PATH` | Live2D 资源目录 | `Live2D` |

---

### 六、数据目录与默认会话

| 参数名 | 说明 | 示例 |
|------|------|------|
| `DATA_TTS_DIR` | TTS 数据输出目录 | `data/tts` |
| `DATA_LIVE_SESSIONS_DIR` | 直播会话数据目录 | `data/live_sessions` |
| `DEFAULT_ROOM` | 默认房间名 | `default_room` |
| `DEFAULT_SENDER_NAME` | 默认用户名称 | `路人甲` |

---

### 七、主播人格 / 知识库（Smart Anchor）

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

### 八、流式输出与 TTS 行为

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

### 九、情绪系统（Emotion → TTS / Live2D）

#### 基础配置

| 参数名 | 说明 | 示例 |
|------|------|------|
| `EMOTION_ENABLED` | 是否启用情绪系统 | `1` |
| `EMOTION_LABELS` | 支持的情绪标签 | 正常,开心,失落,疑惑,惊讶,生气 |
| `EMOTION_DEFAULT` | 默认情绪 | 正常 |
| `EMOTION_SYSTEM_PROMPT` | 情绪输出规则 | 回复末尾必须包含情绪标签 |

#### Live2D 动作映射

| 参数名 | 说明 | 示例 |
|------|------|------|
| `EMOTION_MAP_正常` | 正常情绪动作 | `Tap,0` |
| `EMOTION_MAP_开心` | 开心情绪动作 | `Tap,1\|FlickUp,0` |
| `EMOTION_MAP_失落` | 失落情绪动作 | `Flick@Body,0` |
| `EMOTION_MAP_疑惑` | 疑惑情绪动作 | `Flick,0` |
| `EMOTION_MAP_惊讶` | 惊讶情绪动作 | `FlickDown,0` |
| `EMOTION_MAP_生气` | 生气情绪动作 | `Tap@Body,0` |

#### 调试参数

| 参数名 | 说明 | 示例 |
|------|------|------|
| `DEBUG_TTS` | TTS 调试日志 | `1` |
| `DEBUG_EMOTION` | 情绪调试日志 | `1` |

---

### 十、健康检查

| 参数名 | 说明 | 示例 |
|------|------|------|
| `HEALTH_CHECK_TIMEOUT_SECONDS` | 健康检查超时时间（秒） | `5` |

---

> 建议：  
> - **生产环境**关闭 `DEBUG_*` 相关参数  
> - **低显存设备**合理降低 `SESSION_LEN` 与并发数  
> - **直播场景**建议开启流式输出与情绪系统
## 未来计划
 - [ ]   ~~（学不会）~~  学编程
 - [ ] （优先）接入真正的直播系统
 - [ ] （优先）增加TTS的情感模块
 - [ ] （优先）改进Live2d模型适配
 - [ ] 改进口型处理
 - [ ] 接入LLMapi和TTSapi
 - [ ] 训练虚拟主播专用微调模型
 - [ ] 美化webui
 - [ ] 接入互联网搜索
 - [ ] 让AI操作电脑
 - [ ] 其他
## 联系我
~~联系我也没用，我又不会编程~~
QQ：1792767935
gmail：icjztdhop@gmail.com
## 致谢
感谢[LMDeploy](https://github.com/InternLM/lmdeploy/)
感谢[书生·浦语](https://internlm.intern-ai.org.cn/)
感谢[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
感谢[pixi](https://pixijs.com/)
感谢[LlamaIndex](https://github.com/run-llama/llama_index)
感谢[faiss](https://github.com/facebookresearch/faiss)
感谢其他python开源库
## 开源许可
允许个人使用，更改
禁止企业商用  ~~（未达到商用级别）~~
**搬运标明来源**
Licensed under the Apache License, Version 2.0.
See the [LICENSE](https://oozie.apache.org/license.html) for details.



