#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_server.py

FastAPI backend for a single virtual anchor:
- /chat_once: JSON
- /chat_stream: SSE streaming (delta / segment / audio)
- /ws: WebSocket streaming (delta / segment / audio)
- /tts_once: one-shot TTS

Notes:
- Emotion tags (【emo=...】) are used for animation trigger, not for bubble text.
- Streaming mode filters emotion/reference tags from delta, while keeping emotion info on segment/audio events.
- ✅ LiveFullLog records FULL user+assistant transcript (never trimmed). Persisted on shutdown/CTRL+C.
"""

from __future__ import annotations

import os
import time
import json
import re
import asyncio
import threading
import webbrowser
import traceback
import signal
import atexit
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Iterable, Callable
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from smart_anchor import SmartVirtualAnchor, load_config, cfg_bool, resolve_config_path
from contextlib import asynccontextmanager

def _env_str(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    return default if v is None else str(v)

def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    try:
        return int(v) if v is not None and str(v).strip() != "" else default
    except Exception:
        return default

def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_csv(key: str, default: str = "") -> list[str]:
    s = _env_str(key, default)
    if not s:
        return []
    # 支持 "*" 或 "a,b,c" 或 "a；b；c"
    s = s.replace("，", ",").replace("；", ";")
    return [x.strip() for x in re.split(r"[,;]", s) if x.strip()]

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _env_bool("DISABLE_AUTO_OPEN", False):
        port = _env_int("API_PORT", 8000)
        base = _env_str("API_PUBLIC_BASE_URL", f"http://127.0.0.1:{port}")
        url = base.rstrip("/") + "/web/"

        def _open():
            delay = _env_int("AUTO_OPEN_DELAY_MS", 600) / 1000.0
            time.sleep(delay)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    try:
        yield
    finally:
        persist_all_live_logs(reason="lifespan_shutdown")


# ----------------------------
# Tag regex
# ----------------------------
EMO_TAG_RE = re.compile(r"【emo=([^】\]]+)】")
REF_TAG_RE = re.compile(r"【参考资料[:：][^】]*】")
SEQ_RE = re.compile(r"^\s*#\d+\s*")
DEBUG_LLM_STREAM = _env_bool("DEBUG_LLM_STREAM", False)
DEBUG_LLM_MAX_CHARS = _env_int("DEBUG_LLM_MAX_CHARS", 200)
SENT_END_RE = re.compile(r"[。！？!?]\s*$")
_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)


class LLMConsoleJoiner:
    def __init__(self, tag: str, max_preview: int = 4000) -> None:
        self.tag = tag
        self.buf = ""
        self.max_preview = max_preview

    def push(self, delta: str) -> None:
        if delta:
            self.buf += delta

    def maybe_flush(self) -> None:
        s = self.buf.strip()
        if not s:
            return
        if SENT_END_RE.search(s):
            out = s
            if len(out) > self.max_preview:
                out = out[-self.max_preview :]
            print(f"[LLM_FULL]{self.tag} {out}")
            self.buf = ""

    def flush_all(self) -> None:
        s = self.buf.strip()
        if not s:
            return
        out = s
        if len(out) > self.max_preview:
            out = out[-self.max_preview :]
        print(f"[LLM_FULL]{self.tag} {out}")
        self.buf = ""


def remove_exact_sender_mention(text: str, sender_name: Optional[str]) -> str:
    s = (text or "").lstrip()
    if not sender_name:
        return s
    sender_name = sender_name.strip()
    prefix = f"@{sender_name}"
    if not s.startswith(prefix):
        return s
    s = s[len(prefix) :]
    s = re.sub(r"^[\s,，]+", "", s)
    return s


def remove_leading_mention(text: str) -> str:
    s = (text or "").lstrip()
    if not s.startswith("@"):
        return s
    s2 = s[1:]
    m = re.match(r"([^\s,，]{1,32})", s2)
    if not m:
        return s
    name = m.group(1)
    rest = s2[len(name) :]
    rest = re.sub(r"^[\s,，]+", "", rest)
    return rest


def stream_chunk_to_delta(chunk: str, last_cum: str) -> tuple[str, str]:
    chunk = chunk or ""
    last_cum = last_cum or ""

    # 典型累计流
    if chunk.startswith(last_cum):
        return chunk[len(last_cum) :], chunk

    # 可能发生回退/重置（chunk 更短）
    if len(chunk) < len(last_cum) and chunk and (last_cum.find(chunk) == 0):
        return chunk, chunk

    # 否则按增量流处理：delta = chunk，累计拼起来
    return chunk, last_cum + chunk

# ----------------------------
# App
# ----------------------------
app = FastAPI(
    title="Virtual Anchor API",
    description="Single-anchor backend: LLM + TXT knowledge base + optional TTS (streaming, chunked audio)",
    version="1.0.0",
    lifespan=lifespan,
)

allow_origins = _env_str("CORS_ALLOW_ORIGINS", "*")
origins = ["*"] if not allow_origins.strip() or allow_origins.strip() == "*" else _env_csv("CORS_ALLOW_ORIGINS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=_env_bool("CORS_ALLOW_CREDENTIALS", True),
    allow_methods=["*"] if _env_str("CORS_ALLOW_METHODS", "*").strip() == "*" else _env_csv("CORS_ALLOW_METHODS"),
    allow_headers=["*"] if _env_str("CORS_ALLOW_HEADERS", "*").strip() == "*" else _env_csv("CORS_ALLOW_HEADERS"),
)

BASE_DIR = Path(__file__).resolve().parent
web_dir = BASE_DIR / _env_str("STATIC_WEB_PATH", "web")
live2d_dir = BASE_DIR / _env_str("STATIC_LIVE2D_PATH", "Live2D")

app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")
app.mount("/Live2D", StaticFiles(directory=str(live2d_dir)), name="Live2D")


##@app.on_event("startup")
##def _open_browser_on_startup():
##    if os.environ.get("DISABLE_AUTO_OPEN", "0") == "1":
##        return
##    port = int(os.environ.get("PORT", "8000"))
##    url = f"http://127.0.0.1:{port}/web/"
##
##    def _open():
##        time.sleep(0.6)
##        try:
##            webbrowser.open(url)
##        except Exception:
##            pass
##
##    threading.Thread(target=_open, daemon=True).start()
##
# ----------------------------
# Anchor (GLOBAL, MUST EXIST)
# ----------------------------
_lm_host = _env_str("LMDEPLOY_HOST", "127.0.0.1")
_lm_port = _env_int("LMDEPLOY_PORT", 23333)
_lm_url = _env_str("LMDEPLOY_URL", f"http://{_lm_host}:{_lm_port}")

ANCHOR = SmartVirtualAnchor(
    lmdeploy_url=_lm_url,
    name=_env_str("ANCHOR_NAME", "小爱"),
    knowledge_txt_path=_env_str("KNOWLEDGE_TXT", "knowledge.txt"),
)

TTS_DIR = _env_str("DATA_TTS_DIR", "data/tts")
os.makedirs(TTS_DIR, exist_ok=True)

LIVE_SESS_DIR = _env_str("DATA_LIVE_SESSIONS_DIR", "data/live_sessions")
os.makedirs(LIVE_SESS_DIR, exist_ok=True)

# ----------------------------
# Live session full log (NOT trimmed)
# ----------------------------
@dataclass
class LiveFullLog:
    started_at: float = field(default_factory=time.time)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # full, never trimmed

    def add(self, role: str, content: str, sender_name: Optional[str] = None) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "sender_name": sender_name,
                "ts": time.time(),
            }
        )

    def to_transcript(self, anchor_name: str) -> str:
        lines: List[str] = []
        for m in self.messages:
            role = m.get("role")
            c = (m.get("content") or "").strip()
            if not c:
                continue
            if role == "user":
                sn = m.get("sender_name") or "观众"
                lines.append(f"{sn}：{c}")
            else:
                lines.append(f"{anchor_name}：{c}")
        return "\n".join(lines).strip()


LIVE_LOGS: Dict[str, LiveFullLog] = {}  # user_id -> LiveFullLog
_LIVE_PERSISTED = False


def persist_all_live_logs(reason: str = "shutdown") -> None:
    global _LIVE_PERSISTED
    if _LIVE_PERSISTED:
        return
    _LIVE_PERSISTED = True

    ended_at = time.time()

    for user_id, log in list(LIVE_LOGS.items()):
        if not log.messages:
            continue

        started_at = log.started_at
        tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
        txt_path = os.path.join(LIVE_SESS_DIR, f"live_full_{user_id}_{tag}.txt")
        json_path = os.path.join(LIVE_SESS_DIR, f"live_full_{user_id}_{tag}.json")
        transcript = log.to_transcript(anchor_name=ANCHOR.anchor_name)

        # 1) write files
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"【直播结束落库】reason={reason}\n")
                f.write(f"started_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}\n")
                f.write(f"ended_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ended_at))}\n\n")
                f.write(transcript + "\n")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(log.messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[PERSIST] write files failed user={user_id} err={e}")

        # 2) write to RAG (long-term memory)
        meta = {
            "type": "live_full_log",
            "reason": reason,
            "anchor_name": ANCHOR.anchor_name,
            "user_id": user_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": round(ended_at - started_at, 2),
            "txt_path": txt_path,
            "json_path": json_path,
        }

        rag_text = "\n".join(
            [
                f"【完整直播记录】anchor={ANCHOR.anchor_name} user_id={user_id}",
                f"started_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}",
                f"ended_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ended_at))}",
                "",
                "【完整对话转写】",
                transcript,
            ]
        ).strip()

        try:
            mid = ANCHOR.add_long_term_memory(user_id, rag_text, metadata=meta)
            print(f"[PERSIST] RAG saved ✓ user={user_id} memory_id={mid}")
        except Exception as e:
            print(f"[PERSIST] RAG save failed ✗ user={user_id} err={e}")


def _signal_handler(signum, frame):
    try:
        persist_all_live_logs(reason=f"signal_{signum}")
    finally:
        raise KeyboardInterrupt


try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    pass

atexit.register(lambda: persist_all_live_logs(reason="atexit"))

# ----------------------------
# Config helpers
# ----------------------------
def _split_csv(s: str) -> List[str]:
    if not s:
        return []
    s = str(s).replace("，", ",").replace("；", ";")
    return [x.strip() for x in re.split(r"[,;]", s) if x.strip()]


def _parse_emotion_motion_map(cfg: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in (cfg or {}).items():
        if not k.startswith("EMOTION_MAP_"):
            continue
        emo = k[len("EMOTION_MAP_") :].strip()
        raw = (v or "").strip()
        if not emo or not raw:
            continue

        opts: List[Dict[str, Any]] = []
        for item in raw.split("|"):
            item = item.strip()
            if not item:
                continue
            if "," in item:
                g, i = item.split(",", 1)
                g = g.strip()
                try:
                    idx = int(i.strip())
                except Exception:
                    idx = 0
                if g:
                    opts.append({"group": g, "index": idx})
            else:
                g = item.strip()
                if g:
                    opts.append({"group": g, "index": 0})

        if opts:
            out[emo] = opts
    return out


def get_runtime_config() -> Dict[str, Any]:
    cfg_path = resolve_config_path()
    cfg = load_config(cfg_path)

    emotion_enabled = cfg_bool(cfg, "EMOTION_ENABLED", False)
    emotion_labels = _split_csv(cfg.get("EMOTION_LABELS", ""))
    emotion_motion_map = _parse_emotion_motion_map(cfg) if emotion_enabled else {}

    return {
        "config_path": cfg_path,
        "stream_llm": cfg_bool(cfg, "STREAM_LLM", False),
        "stream_tts_chunked": cfg_bool(cfg, "STREAM_TTS_CHUNKED", True),
        "tts_workers": int(cfg.get("TTS_WORKERS", "2") or "2"),
        "max_pending_chars": int(cfg.get("MAX_PENDING_CHARS", "160") or "160"),
        "min_tts_chars": int(cfg.get("MIN_TTS_CHARS", "8") or "8"),
        "flush_max_segments": int(cfg.get("FLUSH_MAX_SEGMENTS", "2") or "2"),
        "poll_interval_ms": int(cfg.get("POLL_INTERVAL_MS", "10") or "10"),
        "debug_tts": cfg_bool(cfg, "DEBUG_TTS", False),
        "emotion_enabled": emotion_enabled,
        "emotion_labels": emotion_labels,
        "emotion_motion_map": emotion_motion_map,
        "debug_emotion": cfg_bool(cfg, "DEBUG_EMOTION", False),
    }


def ensure_emo_tag(text: str, cfg: Dict[str, Any]) -> str:
    s = (text or "").rstrip()
    if not cfg.get("emotion_enabled", False):
        return s

    labels = cfg.get("emotion_labels") or ["正常"]
    default = "正常" if "正常" in labels else labels[0]

    last = None
    for mm in EMO_TAG_RE.finditer(s):
        last = mm
    if last:
        emo = (last.group(1) or "").strip()
        if emo not in labels:
            s = EMO_TAG_RE.sub(f"【emo={default}】", s)
        return s

    return s + f"【emo={default}】"


# ----------------------------
# Request models
# ----------------------------
class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    sender_name: Optional[str] = None
    message: str
    tts: bool = False
    tts_params: Optional[Dict[str, Any]] = None


class TTSOnceRequest(BaseModel):
    text: str
    sender_name: Optional[str] = None
    tts_params: Optional[Dict[str, Any]] = None


# ----------------------------
# Segmentation / normalization
# ----------------------------
BOUNDARIES = {"，", ",", "。", "！", "!", "？", "?", "；", ";", "\n"}
_STRONG_END_RE = re.compile(r"[。！？!?]\s*$")


def normalize_tts_text(text: str, sender_name: Optional[str] = None) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = EMO_TAG_RE.sub("", s)
    s = REF_TAG_RE.sub("", s)
    s = SEQ_RE.sub("", s)
    s = remove_exact_sender_mention(s, sender_name)
    s = remove_leading_mention(s)
    s = re.sub(r"^[,，、\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    if _PUNCT_ONLY_RE.match(s):
        return ""
    return s


def effective_len_for_threshold(text: str, sender_name: Optional[str] = None) -> int:
    s = normalize_tts_text(text, sender_name=sender_name)
    return len(re.sub(r"[，,。！？!?；;：:\s]+", "", s))


def extract_boundary_segments(buf: str, max_pending_chars: int) -> Tuple[List[str], str]:
    segs: List[str] = []
    start = 0
    for i, ch in enumerate(buf):
        if ch in BOUNDARIES:
            piece = buf[start : i + 1].strip()
            if piece:
                segs.append(piece)
            start = i + 1

    rest = buf[start:].strip()
    if len(rest) > max_pending_chars:
        cut = rest.rfind(" ", 0, max_pending_chars)
        cut = cut if cut > 0 else max_pending_chars
        segs.append(rest[:cut].strip())
        rest = rest[cut:].strip()

    return segs, rest


def merge_tail_to_max_segments(text: str, min_len: int, max_segments: int = 2) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts: List[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in BOUNDARIES:
            piece = text[start : i + 1].strip()
            if piece:
                parts.append(piece)
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)

    merged: List[str] = []
    cur = ""
    for p in parts:
        cur += p
        if len(cur) >= min_len and _STRONG_END_RE.search(cur):
            merged.append(cur.strip())
            cur = ""
        elif len(cur) >= max(min_len * 2, min_len + 20) and len(merged) < max_segments - 1:
            merged.append(cur.strip())
            cur = ""
    if cur.strip():
        merged.append(cur.strip())

    if len(merged) > max_segments:
        head = merged[: max_segments - 1]
        tail2 = "".join(merged[max_segments - 1 :]).strip()
        merged = head + ([tail2] if tail2 else [])

    if len(merged) >= 2 and sum(len(x) for x in merged[:-1]) < min_len:
        merged = ["".join(merged).strip()]

    return [x for x in merged if x.strip()]


def extract_last_emotion(text: str) -> Tuple[str, str, bool]:
    s = (text or "").strip()
    m = None
    for mm in EMO_TAG_RE.finditer(s):
        m = mm
    if not m:
        return s, "", False
    emo = (m.group(1) or "").strip()
    cleaned = (s[: m.start()] + s[m.end() :]).strip()
    return cleaned, emo, True


def _split_by_boundaries_keep_punct(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    for i, ch in enumerate(text or ""):
        if ch in BOUNDARIES:
            piece = (text[start : i + 1] or "").strip()
            if piece:
                parts.append(piece)
            start = i + 1
    tail = (text[start:] or "").strip()
    if tail:
        parts.append(tail)
    return parts


def split_piece_on_emo_tags(text: str, default_emotion: str = "") -> List[Tuple[str, str, bool]]:
    s = text or ""
    out: List[Tuple[str, str, bool]] = []
    last = 0
    found_any = False

    for mm in EMO_TAG_RE.finditer(s):
        found_any = True
        start, end = mm.span()
        emo = (mm.group(1) or "").strip() or default_emotion

        pre = (s[last:start] or "")
        pre = EMO_TAG_RE.sub("", pre)
        pre = REF_TAG_RE.sub("", pre).strip()

        if pre:
            pre_parts = _split_by_boundaries_keep_punct(pre)
            for p in pre_parts[:-1]:
                p = (p or "").strip()
                if p:
                    out.append((p, default_emotion, False))
            last_part = (pre_parts[-1] or "").strip()
            if last_part:
                out.append((last_part, emo, True))

        last = end

    tail = (s[last:] or "")
    tail = EMO_TAG_RE.sub("", tail)
    tail = REF_TAG_RE.sub("", tail).strip()
    for p in _split_by_boundaries_keep_punct(tail):
        p = (p or "").strip()
        if p:
            out.append((p, default_emotion, False))

    if not found_any:
        clean = EMO_TAG_RE.sub("", s)
        clean = REF_TAG_RE.sub("", clean).strip()
        if clean:
            out.append((clean, default_emotion, False))

    return out


class DeltaTagFilter:
    def __init__(self) -> None:
        self._buf = ""
        self._in_bracket = False

    def push(self, delta: str) -> str:
        if not delta:
            return ""
        out = []
        for ch in delta:
            if not self._in_bracket:
                if ch == "【":
                    self._in_bracket = True
                    self._buf = "【"
                else:
                    out.append(ch)
            else:
                self._buf += ch
                if ch == "】":
                    block = self._buf
                    self._buf = ""
                    self._in_bracket = False
                    if EMO_TAG_RE.fullmatch(block) or REF_TAG_RE.fullmatch(block):
                        continue
                    out.append(block)
        return "".join(out)

    def flush(self) -> None:
        self._buf = ""
        self._in_bracket = False


def sse(obj: Dict[str, Any]) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


# ----------------------------
# Streaming engine (shared by WS/SSE)
# ----------------------------
class TTSPool:
    def __init__(self, max_workers: int) -> None:
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: List[Tuple[int, Future, str, str, bool]] = []

    def submit(
        self, seq: int, fn: Callable[[], Dict[str, Any]], display_text: str, emo: str, emo_trigger: bool
    ) -> None:
        fut = self.pool.submit(fn)
        self.futures.append((seq, fut, display_text, emo, emo_trigger))

    def drain_done(self) -> List[Dict[str, Any]]:
        evts: List[Dict[str, Any]] = []
        done_idx = [i for i, (_, fut, *_rest) in enumerate(self.futures) if fut.done()]
        for i in sorted(done_idx, reverse=True):
            s, fut, display_piece, emo, had_tag = self.futures.pop(i)
            try:
                info = fut.result()
            except Exception as e:
                evts.append(
                    {
                        "type": "audio_error",
                        "seq": s,
                        "error": f"future raised: {e}",
                        "text": display_piece,
                        "emotion": emo,
                        "emo_trigger": had_tag,
                    }
                )
                continue

            if info.get("ok") and info.get("filename"):
                evts.append(
                    {
                        "type": "audio",
                        "seq": s,
                        "audio_url": f"/audio/{info['filename']}",
                        "text": display_piece,
                        "emotion": emo,
                        "emo_trigger": had_tag,
                    }
                )
            else:
                evts.append(
                    {
                        "type": "audio_error",
                        "seq": s,
                        "error": info.get("error", "tts failed"),
                        "text": display_piece,
                        "emotion": emo,
                        "emo_trigger": had_tag,
                    }
                )
        return evts

    def has_pending(self) -> bool:
        return bool(self.futures)

    def shutdown(self) -> None:
        try:
            self.pool.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass


class StreamEngine:
    def __init__(self, cfg: Dict[str, Any], sender_name: Optional[str], tts_params: Optional[Dict[str, Any]]) -> None:
        self.cfg = cfg
        self.sender_name = sender_name
        self.tts_params = tts_params

        self.join_threshold = max(1, int(cfg["min_tts_chars"]))
        self.max_pending_chars = max(30, int(cfg["max_pending_chars"]))
        self.flush_max_segments = max(1, int(cfg["flush_max_segments"]))
        self.poll_interval = max(1, int(cfg["poll_interval_ms"])) / 1000.0

        self.emo_enabled = bool(cfg.get("emotion_enabled", False))
        self.debug_tts = bool(cfg.get("debug_tts", False))
        self.debug_emo = bool(cfg.get("debug_emotion", False))

        self.delta_filter = DeltaTagFilter() if self.emo_enabled else None

        self.buf = ""
        self.pending_short = ""
        self.seq = 0
        self.emo_seen = False

        self.tts_pool = TTSPool(max_workers=max(1, int(cfg["tts_workers"])))

        # segment concat-dedupe
        self._recent_seg = deque(maxlen=6)  # store recent (text, ts)
        self._concat_dedupe_sec = 2.0  # only check within 2 seconds window

        # short-time identical segment dedupe
        self._last_seg_text = ""
        self._last_seg_time = 0.0

        # last segment info (for tail emo fallback)
        self._last_segment_seq = 0
        self._last_segment_text = ""
        self._last_segment_had_emo = False

        # seq -> latest emotion status (used to override audio events)
        self._seq_emotion: Dict[int, Tuple[str, bool]] = {}

    def _submit_tts(self, clean_piece: str, emo: str, emo_trigger: bool) -> Dict[str, Any]:
        tts_in = normalize_tts_text(clean_piece, sender_name=self.sender_name)
        return ANCHOR.tts_to_file(tts_in, self.tts_params)

    def _emit_segments_from_piece(self, piece: str, default_emotion: str = "") -> List[Tuple[str, str, bool]]:
        if self.emo_enabled:
            return split_piece_on_emo_tags(piece, default_emotion=default_emotion)
        return [(piece, "", False)]

    def _is_concat_duplicate(self, text: str) -> bool:
        now = time.time()
        text = (text or "").strip()
        if not text:
            return False

        def norm(s: str) -> str:
            return re.sub(r"\s+", "", (s or ""))

        target = norm(text)

        recent = [t for (t, ts) in self._recent_seg if (now - ts) <= self._concat_dedupe_sec]
        if len(recent) < 2:
            return False

        max_k = min(6, len(recent))
        for k in range(2, max_k + 1):
            cand = norm("".join(recent[-k:]))
            if cand == target:
                return True
        return False

    def process_delta(self, delta: str, enable_tts: bool) -> Tuple[str, List[Dict[str, Any]]]:
        events: List[Dict[str, Any]] = []
        self.buf += delta

        if self.emo_enabled and "【emo=" in (delta or ""):
            self.emo_seen = True

        delta_out = delta
        if self.delta_filter is not None:
            delta_out = self.delta_filter.push(delta)

        if not enable_tts or not self.cfg.get("stream_tts_chunked", True):
            return delta_out, events

        segs, rest = extract_boundary_segments(self.buf, max_pending_chars=self.max_pending_chars)
        self.buf = rest

        for raw_piece in segs:
            piece = (raw_piece or "").strip()
            if not piece:
                continue

            eff = effective_len_for_threshold(piece, sender_name=self.sender_name)
            if eff < self.join_threshold:
                self.pending_short += piece
                continue

            if self.pending_short:
                piece = (self.pending_short + piece).strip()
                self.pending_short = ""

            for clean_piece, emo, had_tag in self._emit_segments_from_piece(piece):
                clean_piece = (clean_piece or "").strip()
                if not clean_piece:
                    continue

                if _PUNCT_ONLY_RE.match(clean_piece):
                    self.pending_short += clean_piece
                    continue

                if had_tag:
                    self.emo_seen = True

                now = time.time()
                if clean_piece == self._last_seg_text and (now - self._last_seg_time) < 1.0:
                    continue
                self._last_seg_text = clean_piece
                self._last_seg_time = now

                if self._is_concat_duplicate(clean_piece):
                    continue

                self._recent_seg.append((clean_piece, time.time()))

                self.seq += 1
                events.append(
                    {
                        "type": "segment",
                        "seq": self.seq,
                        "text": clean_piece,
                        "emotion": emo,
                        "emo_trigger": had_tag,
                    }
                )

                self._last_segment_seq = self.seq
                self._last_segment_text = clean_piece
                self._last_segment_had_emo = bool(had_tag)

                self._seq_emotion[self.seq] = (emo or "", bool(had_tag))

                if self.debug_emo:
                    print(f"[EMO] segment seq={self.seq} emo={emo!r} trigger={had_tag} text={clean_piece!r}")

                tts_in = normalize_tts_text(clean_piece, sender_name=self.sender_name)
                if self.debug_tts:
                    print(
                        f"[TTS_SUBMIT] seq={self.seq} raw={clean_piece!r} -> tts={tts_in!r} emo={emo!r} trigger={had_tag}"
                    )

                if not tts_in:
                    events.append(
                        {
                            "type": "audio_skip",
                            "seq": self.seq,
                            "reason": "punct_only_or_empty",
                            "text": clean_piece,
                            "emotion": emo,
                            "emo_trigger": had_tag,
                        }
                    )
                    continue

                try:
                    self.tts_pool.submit(
                        self.seq,
                        lambda cp=clean_piece, e=emo, t=had_tag: self._submit_tts(cp, e, t),
                        display_text=clean_piece,
                        emo=emo,
                        emo_trigger=had_tag,
                    )
                except Exception as e:
                    events.append(
                        {
                            "type": "audio_error",
                            "seq": self.seq,
                            "error": f"submit failed: {e}",
                            "text": clean_piece,
                            "emotion": emo,
                            "emo_trigger": had_tag,
                        }
                    )

        return delta_out, events

    def drain_audio_events(self) -> List[Dict[str, Any]]:
        evts = self.tts_pool.drain_done()

        # override audio event emotion/trigger using latest seq emotion state
        for e in evts:
            if e.get("type") == "audio":
                seq = e.get("seq")
                if isinstance(seq, int) and seq in self._seq_emotion:
                    emo, trig = self._seq_emotion[seq]
                    e["emotion"] = emo
                    e["emo_trigger"] = trig

        return evts

    def flush_end(self, enable_tts: bool) -> List[Dict[str, Any]]:
        if self.delta_filter is not None:
            self.delta_filter.flush()

        if not enable_tts or not self.cfg.get("stream_tts_chunked", True):
            return []

        events: List[Dict[str, Any]] = []
        tail_all = (self.pending_short + self.buf).strip()
        self.pending_short = ""
        self.buf = ""

        tail_emo = ""
        tail_has_emo = False
        if self.emo_enabled:
            tail_all, tail_emo, tail_has_emo = extract_last_emotion(tail_all)

        if self.emo_enabled and (not tail_has_emo) and (not self.emo_seen):
            tail_emo = "正常"
            tail_has_emo = True

        if tail_has_emo and self._last_segment_seq > 0 and not self._last_segment_had_emo:
            events.append(
                {
                    "type": "emotion_update",
                    "seq": self._last_segment_seq,
                    "emotion": tail_emo,
                    "emo_trigger": True,
                    "text": self._last_segment_text,
                }
            )
            self._seq_emotion[self._last_segment_seq] = (tail_emo or "", True)
            if self.debug_emo:
                print(
                    f"[EMO] update seq={self._last_segment_seq} emo={tail_emo!r} trigger=True text={self._last_segment_text!r}"
                )
            self.emo_seen = True

        for piece in merge_tail_to_max_segments(
            tail_all, min_len=self.join_threshold, max_segments=self.flush_max_segments
        ):
            default_emo = tail_emo if tail_has_emo else ""
            for clean_piece, emo, had_tag in self._emit_segments_from_piece(piece, default_emotion=default_emo):
                clean_piece = (clean_piece or "").strip()
                if not clean_piece:
                    continue

                if _PUNCT_ONLY_RE.match(clean_piece):
                    continue

                if self._is_concat_duplicate(clean_piece):
                    continue

                self._recent_seg.append((clean_piece, time.time()))

                self.seq += 1
                events.append(
                    {
                        "type": "segment",
                        "seq": self.seq,
                        "text": clean_piece,
                        "emotion": emo,
                        "emo_trigger": had_tag,
                    }
                )

                self._last_segment_seq = self.seq
                self._last_segment_text = clean_piece
                self._last_segment_had_emo = bool(had_tag)
                self._seq_emotion[self.seq] = (emo or "", bool(had_tag))

                if self.debug_emo:
                    print(f"[EMO] flush seq={self.seq} emo={emo!r} trigger={had_tag} text={clean_piece!r}")

                tts_in = normalize_tts_text(clean_piece, sender_name=self.sender_name)
                if not tts_in:
                    continue

                self.tts_pool.submit(
                    self.seq,
                    lambda cp=clean_piece, e=emo, t=had_tag: self._submit_tts(cp, e, t),
                    display_text=clean_piece,
                    emo=emo,
                    emo_trigger=had_tag,
                )

        return events

    async def drain_all_audio_async(self, emit: Callable[[Dict[str, Any]], Any]) -> None:
        while self.tts_pool.has_pending():
            evts = self.drain_audio_events()
            if evts:
                for evt in evts:
                    await emit(evt)
            else:
                await asyncio.sleep(self.poll_interval)

    def drain_all_audio_sync(self) -> Iterable[Dict[str, Any]]:
        while self.tts_pool.has_pending():
            evts = self.drain_audio_events()
            if evts:
                for evt in evts:
                    yield evt
            else:
                time.sleep(self.poll_interval)

    def close(self) -> None:
        self.tts_pool.shutdown()


# ----------------------------
# Routes
# ----------------------------
##@app.on_event("shutdown")
##def _persist_on_shutdown():
##    persist_all_live_logs(reason="fastapi_shutdown")


@app.get("/")
def root():
    return {
        "service": "Virtual Anchor API",
        "status": "running",
        "endpoints": {
            "GET /config": "read runtime config",
            "POST /chat_once": "one-shot JSON",
            "POST /chat_stream": "SSE stream: delta + segment + audio",
            "POST /chat": "auto choose once/stream by config",
            "WS /ws": "WebSocket stream: delta + segment + audio",
            "POST /tts_once": "one-shot TTS",
            "GET /audio/{filename}": "serve wav",
            "GET /health": "health check",
        },
    }


@app.get("/config")
def read_config():
    return get_runtime_config()


@app.post("/chat_once")
def chat_once(req: ChatRequest):
    cfg = get_runtime_config()
    emo_enabled = bool(cfg.get("emotion_enabled", False))

    effective_user_id = (req.user_id or _env_str("DEFAULT_ROOM", "default_room")).strip()
    r = ANCHOR.chat(req.message, user_id=effective_user_id, sender_name=req.sender_name)

    raw_text = ensure_emo_tag(r["response"], cfg)

    if effective_user_id not in LIVE_LOGS:
        LIVE_LOGS[effective_user_id] = LiveFullLog()
    LIVE_LOGS[effective_user_id].add("user", req.message, sender_name=(req.sender_name or r.get("user_name")))
    LIVE_LOGS[effective_user_id].add("assistant", raw_text, sender_name=None)

    clean_text, emo, found = extract_last_emotion(raw_text) if emo_enabled else (raw_text, "", False)

    out: Dict[str, Any] = {
        "user_id": effective_user_id,
        "sender_name": r["user_name"],
        "text": clean_text,
        "processing_time": r["processing_time"],
    }
    if emo_enabled:
        out["emotion"] = emo or ""
        out["emo_trigger"] = bool(found)

    if req.tts:
        tts_in = normalize_tts_text(clean_text, sender_name=req.sender_name)
        if not tts_in:
            out.update({"tts_ok": False, "tts_error": "empty_after_normalize", "audio_url": None})
            return out

        info = ANCHOR.tts_to_file(tts_in, req.tts_params)
        out["tts_ok"] = info.get("ok", False)
        out["tts_error"] = info.get("error")
        out["audio_filename"] = info.get("filename")
        out["audio_url"] = f"/audio/{info['filename']}" if info.get("ok") and info.get("filename") else None

    return out


@app.post("/tts_once")
def tts_once(req: TTSOnceRequest):
    tts_in = normalize_tts_text(req.text, sender_name=req.sender_name)
    if not tts_in:
        return {"ok": False, "error": "empty_after_normalize", "audio_url": None}

    info = ANCHOR.tts_to_file(tts_in, req.tts_params)
    if not info.get("ok"):
        return {"ok": False, "error": info.get("error") or "tts failed", "audio_url": None}
    fn = info.get("filename")
    return {"ok": True, "error": None, "audio_url": f"/audio/{fn}" if fn else None, "audio_filename": fn}


@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = os.path.join(TTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "audio file not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/health")
def health():
    lmdeploy_ok = False
    try:
        import requests as _r

        host = _env_str("LMDEPLOY_HOST", "127.0.0.1")
        port = _env_int("LMDEPLOY_PORT", 23333)
        resp = _r.get(f"http://{host}:{port}/v1/models", timeout=_env_int("HEALTH_CHECK_TIMEOUT_SECONDS", 2))

        lmdeploy_ok = resp.status_code == 200
    except Exception:
        lmdeploy_ok = False

    return {
        "api_server": "healthy",
        "lmdeploy": "connected" if lmdeploy_ok else "disconnected",
        "timestamp": time.time(),
    }

# ----------------------------
# WebSocket streaming
# ----------------------------
@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    cfg = get_runtime_config()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                req = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({"type": "server_error", "error": "invalid json"}, ensure_ascii=False))
                continue

            user_id = (req.get("user_id") or _env_str("DEFAULT_ROOM", "default_room")).strip()
            sender_name = (req.get("sender_name") or _env_str("DEFAULT_SENDER_NAME", "路人甲")).strip()
            message = (req.get("message") or "").strip()
            enable_tts = bool(req.get("tts", True))
            tts_params = req.get("tts_params")

            if not message:
                await ws.send_text(json.dumps({"type": "server_error", "error": "empty message"}, ensure_ascii=False))
                continue

            speaker_name, it = ANCHOR.chat_stream(
                message,
                user_id=user_id,
                sender_name=sender_name,
            )

            if user_id not in LIVE_LOGS:
                LIVE_LOGS[user_id] = LiveFullLog()
            LIVE_LOGS[user_id].add("user", message, sender_name=sender_name)
            full_raw = ""

            await ws.send_text(
                json.dumps({"type": "meta", "sender_name": speaker_name, "tts": enable_tts}, ensure_ascii=False)
            )

            engine = StreamEngine(cfg=cfg, sender_name=sender_name, tts_params=tts_params)
            llm_joiner = LLMConsoleJoiner(tag="[WS]") if DEBUG_LLM_STREAM else None

            async def emit(evt: Dict[str, Any]) -> None:
                await ws.send_text(json.dumps(evt, ensure_ascii=False))

            try:
                last_cum = ""
                for chunk in it:
                    delta, last_cum = stream_chunk_to_delta(chunk, last_cum)
                    if not delta:
                        continue

                    full_raw += delta

                    if llm_joiner is not None:
                        llm_joiner.push(delta)
                        llm_joiner.maybe_flush()

                    delta_out, seg_events = engine.process_delta(delta, enable_tts=enable_tts)

                    if delta_out:
                        await emit({"type": "delta", "delta": delta_out})

                    if enable_tts:
                        for aevt in engine.drain_audio_events():
                            await emit(aevt)

                    for evt in seg_events:
                        await emit(evt)

                    if enable_tts:
                        for aevt in engine.drain_audio_events():
                            await emit(aevt)

                if llm_joiner is not None:
                    llm_joiner.flush_all()

                for evt in engine.flush_end(enable_tts=enable_tts):
                    await emit(evt)

                if enable_tts:
                    for aevt in engine.drain_audio_events():
                        await emit(aevt)
                    await engine.drain_all_audio_async(emit)

                if full_raw.strip():
                    LIVE_LOGS[user_id].add("assistant", full_raw.strip(), sender_name=None)

                await emit({"type": "done"})

            finally:
                engine.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "server_error", "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        except Exception:
            pass
        traceback.print_exc()


# ----------------------------
# SSE streaming
# ----------------------------
@app.post("/chat_stream")
def chat_stream(req: ChatRequest):
    cfg = get_runtime_config()

    def gen():
        effective_user_id = (req.user_id or _env_str("DEFAULT_ROOM", "default_room")).strip()
        effective_sender = (req.sender_name or _env_str("DEFAULT_SENDER_NAME", "路人甲")).strip()

        speaker_name, it = ANCHOR.chat_stream(
            req.message,
            user_id=effective_user_id,
            sender_name=effective_sender,
        )
        yield sse({"type": "meta", "sender_name": speaker_name, "tts": bool(req.tts)})

        if effective_user_id not in LIVE_LOGS:
            LIVE_LOGS[effective_user_id] = LiveFullLog()
        LIVE_LOGS[effective_user_id].add("user", req.message, sender_name=(effective_sender or speaker_name))
        full_raw = ""

        engine = StreamEngine(cfg=cfg, sender_name=effective_sender, tts_params=req.tts_params)
        llm_joiner = LLMConsoleJoiner(tag="[SSE]") if DEBUG_LLM_STREAM else None

        try:
            last_cum = ""
            for chunk in it:
                delta, last_cum = stream_chunk_to_delta(chunk, last_cum)
                if not delta:
                    continue

                full_raw += delta

                if llm_joiner is not None:
                    llm_joiner.push(delta)
                    llm_joiner.maybe_flush()

                delta_out, seg_events = engine.process_delta(delta, enable_tts=bool(req.tts))

                if delta_out:
                    yield sse({"type": "delta", "delta": delta_out})

                if req.tts:
                    for aevt in engine.drain_audio_events():
                        yield sse(aevt)

                for evt in seg_events:
                    yield sse(evt)

                if req.tts:
                    for aevt in engine.drain_audio_events():
                        yield sse(aevt)

            if llm_joiner is not None:
                llm_joiner.flush_all()

            for evt in engine.flush_end(enable_tts=bool(req.tts)):
                yield sse(evt)

            if req.tts:
                for aevt in engine.drain_audio_events():
                    yield sse(aevt)
                for aevt in engine.drain_all_audio_sync():
                    yield sse(aevt)

            if full_raw.strip():
                LIVE_LOGS[effective_user_id].add("assistant", full_raw.strip(), sender_name=None)

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield sse({"type": "server_error", "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
            yield "data: [DONE]\n\n"
        finally:
            engine.close()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    cfg = get_runtime_config()
    if not cfg.get("stream_llm", False):
        return chat_once(req)
    return chat_stream(req)


if __name__ == "__main__":
    host = _env_str("API_HOST", "0.0.0.0")
    port = _env_int("API_PORT", 8000)
    log_level = _env_str("API_LOG_LEVEL", "info")
    reload_ = _env_bool("API_RELOAD", False)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload_,
        log_level=log_level,
    )