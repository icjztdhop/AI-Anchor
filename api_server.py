#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟主播API服务器 - 严格遵循规则版本

规则:
1. 从前端接收输入
2. 返回给LLM的API服务，开启流式输出
3. LLM流式输出返回给API服务，API开启三个线程处理:
   a. 线程1: 把LLM流式输出返回给前端展示，删除每个delta前面的空格，删除中文大括号内的所有内容
   b. 线程2: 检测流式输出的逗号、句号等分隔符号，以分隔符号为一段发送到TTS服务
      若检测到中文大括号，则等待大括号的括回，舍弃整个大括号范围内的所有内容
   c. 线程3: 情感服务，计算TTS发送给前端数据的次数
      若两次TTS发送还没有检测到中文大括号，则在第三次加入【emo=正常】
      若检测到中文大括号，则匹配config.txt内预设的情感
"""

import os
import sys
import json
import time
import re
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Set, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from live_memory_session import LiveMemoryManager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from smart_anchor import SmartVirtualAnchor, load_config, cfg_bool, resolve_config_path
except ImportError:
    # 模拟实现
    class SmartVirtualAnchor:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "小爱")

        def chat_stream(self, message, user_id=None, sender_name=None):
            responses = [
                "你好啊，",
                "我是虚拟主播【emo=开心】",
                "今天天气不错。",
                "你想聊什么呢？"
            ]

            def stream():
                for part in responses:
                    yield part
                    time.sleep(0.2)

            return self.name, stream()

        def tts_to_file(self, text, params=None):
            import uuid
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            Path("data/tts").mkdir(exist_ok=True)
            with open(f"data/tts/{filename}", "wb") as f:
                f.write(b"")
            return {"ok": True, "filename": filename}

    def load_config(path):
        return {
            "EMOTION_ENABLED": "true",
            "EMOTION_LABELS": "正常,开心,生气,悲伤,惊讶",
            "EMOTION_MAP_正常": "Tap,0",
            "EMOTION_MAP_开心": "Tap,1|FlickUp,0",
            "EMOTION_MAP_生气": "Tap@Body,0",
            "LMDEPLOY_HOST": "127.0.0.1",
            "LMDEPLOY_PORT": "23333"
        }

    def cfg_bool(cfg, key, default):
        val = cfg.get(key, "").lower()
        return val in ("1", "true", "yes", "on") if val else default

    def resolve_config_path():
        return "config.txt"

# ==================== 配置 ====================

def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except:
        return default

def get_env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").lower()
    return val in ("1", "true", "yes", "on") if val else default

# 加载情感配置
def load_emotion_config():
    """加载config.txt中的情感配置"""
    cfg_path = resolve_config_path()
    cfg = load_config(cfg_path)

    emotion_enabled = cfg_bool(cfg, "EMOTION_ENABLED", True)
    emotion_labels = []

    # 解析情感标签
    labels_str = cfg.get("EMOTION_LABELS", "正常,开心,生气,悲伤,惊讶")
    if labels_str:
        emotion_labels = [label.strip() for label in labels_str.split(",") if label.strip()]

    # 解析情感映射
    emotion_map = {}
    for key, value in cfg.items():
        if key.startswith("EMOTION_MAP_"):
            emotion = key.replace("EMOTION_MAP_", "")
            options = []

            for item in str(value).split("|"):
                item = item.strip()
                if not item:
                    continue

                if "," in item:
                    group, idx = item.split(",", 1)
                    options.append({
                        "group": group.strip(),
                        "index": int(idx.strip()) if idx.strip().isdigit() else 0
                    })
                else:
                    options.append({"group": item.strip(), "index": 0})

            if options:
                emotion_map[emotion] = options

    return {
        "enabled": emotion_enabled,
        "labels": emotion_labels,
        "map": emotion_map
    }


_EMOTION_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_EMOTION_CONFIG_CACHE_MTIME: float = 0.0
_EMOTION_CONFIG_LOCK = threading.Lock()


def get_emotion_config_cached() -> Dict[str, Any]:
    """Cache emotion config by file mtime to avoid reloading on every request."""
    global _EMOTION_CONFIG_CACHE, _EMOTION_CONFIG_CACHE_MTIME

    cfg_path = resolve_config_path()
    try:
        mtime = Path(cfg_path).stat().st_mtime
    except Exception:
        mtime = 0.0

    with _EMOTION_CONFIG_LOCK:
        if _EMOTION_CONFIG_CACHE is not None and mtime == _EMOTION_CONFIG_CACHE_MTIME:
            return _EMOTION_CONFIG_CACHE

        cfg = load_emotion_config()
        _EMOTION_CONFIG_CACHE = cfg
        _EMOTION_CONFIG_CACHE_MTIME = mtime
        return cfg

# ==================== 正则表达式 ====================

# 检测中文大括号及其内容
CHINESE_BRACKET_PATTERN = r"【[^】]*】"
# 检测完整的句子分隔符
SENTENCE_DELIMITERS = r"[。！？!?；;，,]"
# 检测空格（开头的空格）
LEADING_SPACE_PATTERN = r"^\s+"

# ==================== 数据结构 ====================

@dataclass
class TTSSegment:
    text: str
    seq: int
    emotion: Optional[str] = None

    def to_dict(self):
        return {
            "text": self.text,
            "seq": self.seq,
            "emotion": self.emotion
        }

@dataclass
class StreamSegment:
    """流式分段"""
    seq: int
    text: str
    emotion: str = ""
    has_emotion_tag: bool = False
    created_at: float = field(default_factory=time.time)
    tts_done: bool = False
    tts_filename: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "text": self.text,
            "emotion": self.emotion,
            "emo_trigger": self.has_emotion_tag
        }

@dataclass
class ChatSession:
    """聊天会话"""
    session_id: str
    user_id: str
    sender_name: str
    created_at: float = field(default_factory=time.time)

    # 状态
    llm_streaming: bool = False
    tts_counter: int = 0  # TTS发送给前端的计数
    last_emotion_detected: float = 0  # 上次检测到情感的时间

    # 缓冲区
    text_buffer: str = ""  # 用于TTS分段的缓冲区
    display_buffer: str = ""  # 用于前端显示的缓冲区
    bracket_buffer: str = ""  # 用于大括号检测的缓冲区
    in_bracket: bool = False  # 是否在大括号内

    # 队列
    tts_queue: Deque[StreamSegment] = field(default_factory=deque)
    display_queue: Deque[str] = field(default_factory=deque)
    emotion_queue: Deque[Dict[str, Any]] = field(default_factory=deque)

    def reset_buffers(self):
        """重置缓冲区"""
        self.text_buffer = ""
        self.display_buffer = ""
        self.bracket_buffer = ""
        self.in_bracket = False

# ==================== 核心处理器 ====================

class StreamProcessor:
    """流式处理器 - 严格遵循规则"""

    def __init__(self, session: ChatSession, config: Dict[str, Any]):
        self.session = session
        self.config = config
        self.emotion_config = get_emotion_config_cached()

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.futures = []

        # 状态
        self.seq_counter = 0
        self.last_tts_seq = 0
        self.emotion_tts_count = 0  # 没有情感标签的TTS计数

        # 锁
        self.lock = threading.Lock()

        # 情感检测
        self.emotion_pattern = re.compile(r"【emo=([^】]+)】", re.I)

    def process_chunk(self, chunk: str) -> Tuple[str, List[StreamSegment], List[Dict[str, Any]]]:
        """
        处理LLM返回的一个chunk
        返回: (显示文本, TTS分段列表, 情感事件列表)
        """
        display_chunks = []
        tts_segments = []
        emotion_events = []

        with self.lock:
            # 规则3a: 线程1 - 前端显示处理
            # 删除开头的空格，删除中文大括号内的所有内容
            display_text = self._process_for_display(chunk)
            if display_text:
                display_chunks.append(display_text)

            # 规则3b: 线程2 - TTS分段处理
            new_segments = self._process_for_tts(chunk)
            tts_segments.extend(new_segments)

            # 规则3c: 线程3 - 情感服务
            new_emotions = self._process_emotions(chunk, new_segments)
            emotion_events.extend(new_emotions)

            return display_text, tts_segments, emotion_events

    def _process_for_display(self, chunk: str) -> str:
        """处理用于前端显示的文本"""
        if not chunk:
            return ""

        # 删除开头的空格
        text = re.sub(LEADING_SPACE_PATTERN, "", chunk)

        # 删除中文大括号内的所有内容
        text = re.sub(CHINESE_BRACKET_PATTERN, "", text)

        return text

    def _process_for_tts(self, chunk: str) -> List[StreamSegment]:
        """处理用于TTS的文本"""
        if not chunk:
            return []

        segments = []

        # 处理字符流
        for char in chunk:
            if self.session.in_bracket:
                # 在大括号内，添加到括号缓冲区
                self.session.bracket_buffer += char

                # 检查是否遇到结束括号
                if char == "】":
                    self.session.in_bracket = False
                    # 清空括号缓冲区（舍弃大括号内的所有内容）
                    self.session.bracket_buffer = ""
            else:
                # 不在大括号内
                if char == "【":
                    # 开始大括号
                    self.session.in_bracket = True
                    self.session.bracket_buffer = char
                else:
                    # 正常字符，添加到文本缓冲区
                    self.session.text_buffer += char

                    # 检查是否为分隔符
                    if re.match(SENTENCE_DELIMITERS, char):
                        # 遇到分隔符，分割为一段
                        if self.session.text_buffer.strip():
                            segment = self._create_tts_segment(self.session.text_buffer)
                            if segment:
                                segments.append(segment)
                            self.session.text_buffer = ""

        # 检查缓冲区是否过长（防止缓冲区无限增长）
        if len(self.session.text_buffer) > 100:
            # 强制分割
            segment = self._create_tts_segment(self.session.text_buffer)
            if segment:
                segments.append(segment)
            self.session.text_buffer = ""

        return segments

    def _create_tts_segment(self, text: str) -> Optional[StreamSegment]:
        if not text:
            return None

        # 删除大括号内容
        clean_text = re.sub(CHINESE_BRACKET_PATTERN, "", text)

        # 删除所有空白字符（包含内部空格）
        clean_text = re.sub(r"\s+", "", clean_text)

        clean_text = clean_text.strip()

        if not clean_text:
            return None

        # 至少包含一个有效字符
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fa5]", clean_text):
            return None

        if len(clean_text) == 1:
            return None

        self.seq_counter += 1
        return StreamSegment(
            seq=self.seq_counter,
            text=clean_text
        )

    def _process_emotions(self, chunk: str, new_segments: List[StreamSegment]) -> List[Dict[str, Any]]:
        """处理情感检测"""
        if not self.emotion_config["enabled"]:
            return []

        events = []

        # 在chunk中查找情感标签
        emotion_matches = list(self.emotion_pattern.finditer(chunk))

        if emotion_matches:
            # 检测到情感标签
            for match in emotion_matches:
                emotion = match.group(1)
                # 检查情感是否在配置中
                if emotion in self.emotion_config["labels"]:
                    # 规则3c: 匹配上则发送情感信息
                    events.append({
                        "type": "emotion_detected",
                        "emotion": emotion,
                        "timestamp": time.time()
                    })

                    # 重置计数
                    self.emotion_tts_count = 0
                    self.session.last_emotion_detected = time.time()

            # 如果有新的TTS分段，为它们标记情感
            for segment in new_segments:
                if emotion_matches:
                    last_emotion = emotion_matches[-1].group(1)
                    if last_emotion in self.emotion_config["labels"]:
                        segment.emotion = last_emotion
                        segment.has_emotion_tag = True
        else:
            # 没有检测到情感标签
            self.emotion_tts_count += len(new_segments)

            # 规则3c: 若两次TTS发送还没有检测到中文大括号，则在第三次加入【emo=正常】
            if self.emotion_tts_count >= 3 and "正常" in self.emotion_config["labels"]:
                # 为当前的分段添加默认情感
                for segment in new_segments:
                    segment.emotion = "正常"
                    segment.has_emotion_tag = True

                events.append({
                    "type": "default_emotion_added",
                    "emotion": "正常",
                    "reason": "连续3次TTS无情感标签",
                    "timestamp": time.time()
                })

                # 重置计数
                self.emotion_tts_count = 0

        return events

    def flush_buffers(self) -> Tuple[List[StreamSegment], List[Dict[str, Any]]]:
        """流结束时清空剩余缓冲区"""
        segments = []
        emotions = []

        # 处理剩余文本缓冲区
        remaining_text = self.session.text_buffer.strip()
        if remaining_text:
            # 删除大括号内容
            clean_text = re.sub(CHINESE_BRACKET_PATTERN, "", remaining_text).strip()

            if clean_text:
                # 如果没有结束标点，补一个句号
                if not re.search(r"[。！？!?；;]$", clean_text):
                    clean_text += "。"

                self.seq_counter += 1
                segment = StreamSegment(
                    seq=self.seq_counter,
                    text=clean_text
                )
                segments.append(segment)

        # 清空缓冲区
        self.session.text_buffer = ""
        self.session.bracket_buffer = ""
        self.session.in_bracket = False

        return segments, emotions

# ==================== TTS管理器 ====================

class TTSManager:
    """TTS管理器"""

    def __init__(self, anchor, max_workers=3):
        self.anchor = anchor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tts_dir = Path(get_env("DATA_TTS_DIR", "data/tts"))
        self.tts_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, segment: StreamSegment) -> Dict[str, Any]:
        """合成TTS"""
        try:
            result = self.anchor.tts_to_file(segment.text)
            if result.get("ok") and result.get("filename"):
                segment.tts_filename = result["filename"]
                segment.tts_done = True

                return {
                    "seq": segment.seq,
                    "audio_url": f"/audio/{result['filename']}",
                    "text": segment.text,
                    "emotion": segment.emotion,
                    "emo_trigger": segment.has_emotion_tag
                }
            else:
                return {
                    "seq": segment.seq,
                    "error": result.get("error", "TTS失败"),
                    "text": segment.text
                }
        except Exception as e:
            return {
                "seq": segment.seq,
                "error": str(e),
                "text": segment.text
            }

    def synthesize_async(self, segment: StreamSegment, callback=None):
        """异步合成TTS"""
        future = self.executor.submit(self.synthesize, segment)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

# ==================== API服务器 ====================

app = FastAPI(
    title="虚拟主播API服务器",
    description="严格遵循规则的虚拟主播后端",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
BASE_DIR = Path(__file__).parent
web_dir = BASE_DIR / get_env("STATIC_WEB_PATH", "web")
live2d_dir = BASE_DIR / get_env("STATIC_LIVE2D_PATH", "Live2D")

if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

if live2d_dir.exists():
    app.mount("/Live2D", StaticFiles(directory=str(live2d_dir)), name="Live2D")

# 全局状态
anchor = SmartVirtualAnchor(
    lmdeploy_url=get_env("LMDEPLOY_URL", f"http://{get_env('LMDEPLOY_HOST', '127.0.0.1')}:{get_env_int('LMDEPLOY_PORT', 23333)}"),
    name=get_env("ANCHOR_NAME", "小爱"),
    knowledge_txt_path=get_env("KNOWLEDGE_TXT", "knowledge.txt")
)

tts_manager = TTSManager(anchor, max_workers=3)
sessions = {}
session_counter = 0

# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    sender_name: Optional[str] = None
    message: str
    tts: bool = True

# ==================== WebSocket处理器 ====================

class WebSocketManager:
    """WebSocket管理器"""

    def __init__(self, websocket):
        self.ws = websocket
        self.session: Optional[ChatSession] = None
        self.processor: Optional[StreamProcessor] = None
        self.live_session_id: Optional[str] = None
        self.live_mgr: Optional[LiveMemoryManager] = None

    async def send_event(self, event_type: str, data: dict):
        """发送事件到前端"""
        event = {"type": event_type, **data}
        # ✅ 临时调试：确认后端确实在发事件（尤其是 start/delta/done）
        if event_type in ("start", "delta", "done", "server_error"):
            print("[WS->UI]", event_type, (event.get("text", "")[:60] if event_type == "delta" else ""))
        await self.ws.send_json(event)

    async def handle_stream(self, request: ChatRequest):
        """处理流式对话（已修复：一条WS连接=一场直播；断开才结束并落盘）"""
        global session_counter

        # ===================== 直播会话：只在第一次消息创建 =====================
        if self.session is None:
            session_counter += 1
            session_id = f"session_{session_counter}"
            self.live_session_id = session_id

            self.session = ChatSession(
                session_id=session_id,
                user_id=request.user_id or "default_room",
                sender_name=request.sender_name or "用户"
            )
            sessions[session_id] = self.session

            # 创建 live memory manager（断开时落盘 live_sessions + 写入 RAG 总结）
            self.live_mgr = LiveMemoryManager(
                anchor=anchor,
                user_id=self.session.user_id,
                sender_name=self.session.sender_name,
                max_ctx_tokens=int(get_env("MAX_CTX_TOKENS", "1800")),
                keep_last=int(get_env("KEEP_LAST", "6")),
                summary_temperature=float(get_env("SUMMARY_TEMPERATURE", "0.2")),
            )
        else:
            session_id = self.live_session_id

        # 加载配置（用于 StreamProcessor）
        cfg_path = resolve_config_path()
        cfg = load_config(cfg_path)

        # ✅ 每条新消息开始前重置缓冲区（非常关键）
        # 否则上一轮的 in_bracket/bracket_buffer/text_buffer 状态会污染下一轮，导致 display_text 永远为空
        self.session.reset_buffers()

        # 创建处理器（每条消息一个新的 processor）
        self.processor = StreamProcessor(self.session, cfg)

        # 用户身份（传给 LLM / 存到记忆里）
        uid = (request.user_id or "").strip()
        sn = (request.sender_name or "").strip()
        llm_sender_name = sn if sn else (f"user_{uid}" if uid else None)

        # 获取LLM流式响应
        speaker_name, llm_stream = anchor.chat_stream(
            request.message,
            user_id=uid,
            sender_name=llm_sender_name
        )

        # 发送元数据
        await self.send_event("meta", {
            "sender_name": speaker_name,
            "tts": request.tts
        })

        # ✅ 不要等到 delta 才 start（否则开头被清洗为""时前端可能一直不渲染）
        await self.send_event("start", {"session_id": session_id, "anchor": anchor.anchor_name})
        first_delta_sent = True
        # ================= 音频异步队列与发送任务 =================
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()
        expected_seq = 1
        audio_buffer = {}
        tts_futures = []

        async def audio_sender():
            nonlocal expected_seq
            while True:
                seq, result = await audio_queue.get()
                if seq == -1:
                    break

                audio_buffer[seq] = result
                while expected_seq in audio_buffer:
                    res = audio_buffer.pop(expected_seq)
                    if "error" in res:
                        await self.send_event("audio_error", res)
                    else:
                        await self.send_event("audio", res)
                    expected_seq += 1

        sender_task = asyncio.create_task(audio_sender())

        # ================= 把同步LLM流变成异步，防止卡死服务 =========
        async def async_llm_generator():
            iterator = iter(llm_stream)

            def safe_next():
                try:
                    return next(iterator)
                except StopIteration:
                    return None

            while True:
                chunk = await loop.run_in_executor(None, safe_next)
                if chunk is None:
                    break
                yield chunk

        # 处理流式输出
        full_display_text = ""
        first_delta_sent = False

        async for chunk in async_llm_generator():
            display_text, tts_segments, emotion_events = self.processor.process_chunk(chunk)

            # 线程1: 发送显示文本到前端
            if display_text:
                full_display_text += display_text
                await self.send_event("delta", {"text": display_text})

            # 线程2: 处理TTS分段
            if request.tts and tts_segments:
                for segment in tts_segments:
                    await self.send_event("segment", segment.to_dict())

                    def tts_callback(result, seg=segment):
                        loop.call_soon_threadsafe(audio_queue.put_nowait, (seg.seq, result))

                    future = tts_manager.synthesize_async(segment, tts_callback)
                    tts_futures.append(future)

            # 线程3: 处理情感事件
            for emotion_event in emotion_events:
                if emotion_event.get("type") == "emotion_detected":
                    await self.send_event("emotion_update", {
                        "emotion": emotion_event["emotion"],
                        "timestamp": emotion_event["timestamp"]
                    })

        # ================= 收尾：flush 最后缓冲 =================
        remaining_segments, remaining_emotions = self.processor.flush_buffers()

        if request.tts and remaining_segments:
            for segment in remaining_segments:
                await self.send_event("segment", segment.to_dict())

                def tts_callback(result, seg=segment):
                    loop.call_soon_threadsafe(audio_queue.put_nowait, (seg.seq, result))

                future = tts_manager.synthesize_async(segment, tts_callback)
                tts_futures.append(future)

        # 等待所有 TTS 线程结束
        for future in tts_futures:
            try:
                await loop.run_in_executor(None, future.result, 15)
            except:
                pass

        # 告诉发送任务结束
        await audio_queue.put((-1, None))
        await sender_task

        # ===================== 关键新增：把“主播最终回复”写入记忆 + live_mgr 全量记录 =====================
        final_reply = (full_display_text or "").strip()

        # 2) 记录主播回复（全量记录）
        if self.live_mgr is not None and final_reply:
            self.live_mgr.live.add_chunk(f"主播「{anchor.anchor_name}」：{final_reply}")

        # 3) 写入 anchor session memory（chat_stream 不会写 assistant）
        try:
            if final_reply:
                uid2 = request.user_id or "default_room"
                anchor_session = anchor._get_session(uid2)
                anchor_session.memory.add_message("assistant", final_reply)
                anchor_session.memory.trim_keep_last(anchor.memory_keep_last)

                # 可选：每轮都做一次 compact 检查（让 live_chunks 会逐步生成摘要）
                if self.live_mgr is not None:
                    self.live_mgr.maybe_compact_context()
        except Exception as e:
            print(f"[LIVE] append assistant to memory failed: {e}")

        # 发送完成事件
        await self.send_event("done", {})

        # ⚠️ 不要在这里 del sessions[session_id]！
        # 因为你要求“WebSocket断开才算直播结束”，断开时还需要 live_session_id 对应的会话。
        # 清理会话放到 websocket_endpoint 的 WebSocketDisconnect 里统一做。

# ==================== API端点 ====================

@app.get("/")
async def root():
    return {
        "service": "虚拟主播API",
        "version": "1.0.0",
        "status": "running",
        "rules": [
            "1. 从前端接收输入",
            "2. 流式调用LLM API",
            "3. 三线程处理: 显示/TTS/情感"
        ]
    }

@app.get("/config")
async def get_config():
    """获取配置"""
    cfg_path = resolve_config_path()
    cfg = load_config(cfg_path)

    emotion_map = {}
    for key, value in cfg.items():
        if key.startswith("EMOTION_MAP_"):
            emotion = key.replace("EMOTION_MAP_", "")
            options = []

            for item in str(value).split("|"):
                if "," in item:
                    group, idx = item.split(",", 1)
                    options.append({
                        "group": group.strip(),
                        "index": int(idx.strip()) if idx.strip().isdigit() else 0
                    })
                else:
                    options.append({"group": item.strip(), "index": 0})

            if options:
                emotion_map[emotion] = options

    return {
        "emotion_enabled": cfg_bool(cfg, "EMOTION_ENABLED", True),
        "emotion_labels": [l.strip() for l in cfg.get("EMOTION_LABELS", "正常,开心,生气,悲伤,惊讶").split(",") if l.strip()],
        "emotion_motion_map": emotion_map,
        "stream_llm": cfg_bool(cfg, "STREAM_LLM", True),
        "stream_tts_chunked": cfg_bool(cfg, "STREAM_TTS_CHUNKED", True)
    }

@app.post("/chat_once")
async def chat_once(request: ChatRequest):
    """单次对话（兼容性端点）"""
    uid = (request.user_id or "").strip()
    sn = (request.sender_name or "").strip()

    if uid:
        llm_sender_name = sn if sn else f"user_{uid}"
    else:
        llm_sender_name = sn if sn else None

    response = anchor.chat(
        request.message,
        user_id=uid,
        sender_name=llm_sender_name
    )

    result = {
        "user_id": request.user_id or "default_room",
        "sender_name": response.get("user_name", "主播"),
        "text": response["response"],
        "processing_time": response.get("processing_time", 0)
    }

    if request.tts:
        tts_result = anchor.tts_to_file(response["response"])
        if tts_result.get("ok"):
            result["audio_url"] = f"/audio/{tts_result['filename']}"
            result["tts_ok"] = True
        else:
            result["tts_ok"] = False
            result["tts_error"] = tts_result.get("error")

    return result

@app.post("/tts_once")
async def tts_once(text: str):
    """单次TTS"""
    tts_result = anchor.tts_to_file(text)
    if tts_result.get("ok"):
        return {
            "ok": True,
            "audio_url": f"/audio/{tts_result['filename']}",
            "filename": tts_result['filename']
        }
    else:
        return {
            "ok": False,
            "error": tts_result.get("error", "TTS失败")
        }

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取音频文件"""
    filepath = Path("data/tts") / filename
    if not filepath.exists():
        raise HTTPException(404, "音频文件不存在")

    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=filename
    )

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions": len(sessions)
    }

# ==================== WebSocket端点 ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket流式对话 - 严格遵循规则"""
    await websocket.accept()

    manager = WebSocketManager(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            try:
                request = ChatRequest(**data)
            except Exception as e:
                await manager.send_event("server_error", {"error": f"请求格式错误: {e}"})
                continue

            if not request.message.strip():
                await manager.send_event("server_error", {"error": "消息不能为空"})
                continue

            await manager.handle_stream(request)

    except WebSocketDisconnect:
        print("WebSocket连接断开")

        # 断开即视为直播结束：落盘 live_sessions + 写入 RAG 总结
        try:
            if manager.live_mgr is not None:
                res = manager.live_mgr.end_live_and_persist(metadata={
                    "session_id": manager.live_session_id,
                    "disconnect_ts": time.time(),
                })
                print("[LIVE] end_live_and_persist:", json.dumps(res, ensure_ascii=False))
        except Exception as e:
            print("[LIVE] end_live_and_persist failed:", e)

        # 清理会话（这里才删）
        try:
            if manager.live_session_id and manager.live_session_id in sessions:
                del sessions[manager.live_session_id]
        except:
            pass

    except Exception as e:
        print("WebSocket错误:", e)
        import traceback
        traceback.print_exc()
        try:
            await manager.send_event("server_error", {"error": str(e)})
        except:
            pass

# ==================== SSE端点 ====================

@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """SSE流式对话"""

    async def event_generator():
        class MockWebSocket:
            async def send_json(self, data):
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        mock_ws = MockWebSocket()
        manager = WebSocketManager(mock_ws)

        try:
            await manager.handle_stream(request)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'server_error', 'error': str(e)}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# ==================== 主程序 ====================

if __name__ == "__main__":
    host = get_env("API_HOST", "0.0.0.0")
    port = get_env_int("API_PORT", 8000)

    print(f"=== 虚拟主播API服务器 ===")
    print(f"规则:")
    print(f"1. 从前端接收输入")
    print(f"2. 流式调用LLM API")
    print(f"3. 三线程处理: 显示/TTS/情感")
    print(f"服务器启动在: http://{host}:{port}")
    print(f"前端页面: http://{host}:{port}/web/")
    print(f"WebSocket端点: ws://{host}:{port}/ws")
    print("=" * 30)

    Path("data/tts").mkdir(parents=True, exist_ok=True)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )