# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import os
import time
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple

import requests

from txt_knowledge import TxtKnowledgeBase


# ----------------------------
# config.txt loader (single source of truth)
# ----------------------------
def load_config(path: str) -> Dict[str, str]:
    """
    Read KEY=VALUE lines from config.txt (UTF-8).
    - Ignore blank lines
    - Ignore lines starting with # or ;
    - Only split on first '='
    """
    cfg: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return cfg

    text = p.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            cfg[k] = v
    return cfg


def env_or_cfg(cfg: Dict[str, str], key: str, default: str = "") -> str:
    """
    Prefer environment variable, fallback to config.txt, then default.
    """
    v = os.environ.get(key, "").strip()
    if v:
        return v
    return (cfg.get(key, default) or "").strip()


def cfg_bool(cfg: Dict[str, str], key: str, default: bool = False) -> bool:
    v = (env_or_cfg(cfg, key, "") or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def cfg_int(cfg: Dict[str, str], key: str, default: int) -> int:
    v = (env_or_cfg(cfg, key, "") or "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def cfg_float(cfg: Dict[str, str], key: str, default: float) -> float:
    v = (env_or_cfg(cfg, key, "") or "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _split_csv(s: str) -> List[str]:
    if not s:
        return []
    # allow Chinese comma/semicolon and common separators
    s = s.replace("，", ",").replace("；", ";")
    parts: List[str] = []
    for chunk in re.split(r"[,;]", s):
        t = chunk.strip()
        if t:
            parts.append(t)
    return parts


def cfg_list(cfg: Dict[str, str], key: str, default: str = "") -> List[str]:
    raw = (env_or_cfg(cfg, key, "") or "").strip()
    if not raw:
        raw = default
    return _split_csv(raw)


def resolve_config_path() -> str:
    """
    Priority:
    1) env CONFIG_FILE
    2) project root "config.txt" (same dir as this file)
    """
    env_path = os.environ.get("CONFIG_FILE", "").strip()
    if env_path:
        return env_path

    here = Path(__file__).resolve().parent
    default_path = here / "config.txt"
    return str(default_path)


def _resolve_path_maybe_relative(path_str: str, base_dir: str) -> str:
    if not path_str:
        return path_str
    try:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str((Path(base_dir) / p).resolve())
    except Exception:
        return path_str


@dataclass
class ConversationMemory:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    rolling_summary: str = ""

    def add_message(self, role: str, content: str, sender_name: Optional[str] = None) -> None:
        self.messages.append({"role": role, "content": content, "sender_name": sender_name, "ts": time.time()})

    def trim_keep_last(self, keep_last: int) -> None:
        if keep_last > 0 and len(self.messages) > keep_last:
            self.messages = self.messages[-keep_last:]

    def get_recent_context(self, max_messages: int = 6) -> List[Dict[str, Any]]:
        if max_messages <= 0:
            return []
        return self.messages[-max_messages:]


@dataclass
class SessionState:
    user_id: str
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    rag: Optional["RAGMemoryManager"] = None
    last_sender_name: str = "观众"


class SmartVirtualAnchor:
    def __init__(
        self,
        lmdeploy_url: str = "http://localhost:23333",
        name: str = "小爱",
        personality: Optional[Dict[str, Any]] = None,
        knowledge_txt_path: str = "knowledge.txt",
        config_file: Optional[str] = None,
    ) -> None:
        # Load config
        cfg_path = config_file or resolve_config_path()
        self.cfg_path = cfg_path
        self.cfg = load_config(cfg_path)
        cfg_dir = Path(cfg_path).resolve().parent

        # ---------- defaults from config ----------
        # default session
        self.default_user_id = env_or_cfg(self.cfg, "DEFAULT_ROOM", "default_room").strip() or "default_room"
        self.default_sender_name = env_or_cfg(self.cfg, "DEFAULT_SENDER_NAME", "观众").strip() or "观众"

        # LLM endpoint: prefer LMDEPLOY_URL, otherwise build from LMDEPLOY_HOST/PORT
        lmdeploy_url_cfg = env_or_cfg(self.cfg, "LMDEPLOY_URL", "").strip()
        if not lmdeploy_url_cfg:
            host = env_or_cfg(self.cfg, "LMDEPLOY_HOST", "127.0.0.1").strip() or "127.0.0.1"
            port = env_or_cfg(self.cfg, "LMDEPLOY_PORT", "23333").strip() or "23333"
            lmdeploy_url_cfg = f"http://{host}:{port}"
        lmdeploy_url = env_or_cfg(self.cfg, "LMDEPLOY_URL", lmdeploy_url_cfg) or lmdeploy_url_cfg

        # anchor name
        name = env_or_cfg(self.cfg, "ANCHOR_NAME", name)

        # knowledge txt path
        knowledge_txt_path = env_or_cfg(self.cfg, "KNOWLEDGE_TXT", knowledge_txt_path)
        knowledge_txt_path = _resolve_path_maybe_relative(knowledge_txt_path, str(cfg_dir))

        self.lmdeploy_url = (lmdeploy_url or "").rstrip("/")
        self.anchor_name = name
        self.sessions: Dict[str, SessionState] = {}

        # ---------- HTTP / timeout ----------
        # Use HEALTH_CHECK_TIMEOUT_SECONDS as a base default if present
        health_timeout = cfg_int(self.cfg, "HEALTH_CHECK_TIMEOUT_SECONDS", 5)

        # If your config doesn't define these, they fallback to good defaults
        self.http_timeout_models = cfg_int(self.cfg, "HTTP_TIMEOUT_MODELS", 3)
        self.http_timeout_chat = cfg_int(self.cfg, "HTTP_TIMEOUT_CHAT", 30)
        self.http_timeout_stream = cfg_int(self.cfg, "HTTP_TIMEOUT_STREAM", 300)
        self.http_timeout_tts = cfg_int(self.cfg, "HTTP_TIMEOUT_TTS", 120)

        # allow using health timeout to override everything if you want (optional)
        if cfg_bool(self.cfg, "HTTP_TIMEOUT_FOLLOW_HEALTH", False):
            self.http_timeout_models = health_timeout
            self.http_timeout_chat = max(health_timeout, 10)
            self.http_timeout_stream = max(health_timeout, 60)
            self.http_timeout_tts = max(health_timeout, 30)

        self._http = requests.Session()
        self._http.headers.update({"Connection": "keep-alive"})

        # ---------- Debug ----------
        # Enable by env DEBUG_SHOW_LLM_INPUT=1 or config DEBUG_SHOW_LLM_INPUT=true
        self.debug_show_llm_input = cfg_bool(self.cfg, "DEBUG_SHOW_LLM_INPUT", False) or (
            os.environ.get("DEBUG_SHOW_LLM_INPUT", "").strip().lower() in ("1", "true", "yes", "on")
        )
        # Avoid printing overly huge prompt (prints head+tail if too long)
        self.debug_prompt_max_chars = cfg_int(self.cfg, "DEBUG_PROMPT_MAX_CHARS", 6000)

        # Model name cache
        self._model_name_cache: str = ""
        self._model_name_cache_ts: float = 0.0
        self._model_name_cache_ttl = cfg_float(self.cfg, "MODEL_NAME_CACHE_TTL", 300.0)

        # ---------- LLM generation params ----------
        self.llm_temperature = cfg_float(self.cfg, "ANCHOR_TEMPERATURE", 0.7)
        self.llm_top_p = cfg_float(self.cfg, "ANCHOR_TOP_P", 0.9)
        self.llm_max_tokens = cfg_int(self.cfg, "ANCHOR_MAX_TOKENS", 300)

        # Output length control (matches your config key)
        self.anchor_max_response_length = cfg_int(self.cfg, "ANCHOR_MAX_RESPONSE_LENGTH", 500)

        # stream behavior (matches your config key)
        self.stream_llm_enabled = cfg_bool(self.cfg, "STREAM_LLM", True)

        # ---------- Memory / history behavior ----------
        self.memory_keep_last = cfg_int(self.cfg, "MEMORY_KEEP_LAST", 20)
        self.prompt_history_max_messages = cfg_int(self.cfg, "PROMPT_HISTORY_MAX_MESSAGES", 6)

        # ---------- Knowledge base behavior ----------
        self.kb_top_k = cfg_int(self.cfg, "KB_TOP_K", 4)
        self.kb_score_threshold_prompt = cfg_float(self.cfg, "KB_SCORE_THRESHOLD_PROMPT", 0.35)
        self.kb_score_threshold_chat = cfg_float(self.cfg, "KB_SCORE_THRESHOLD_CHAT", 0.18)
        self.kb_force_grounded_on_hits = cfg_bool(self.cfg, "KB_FORCE_GROUNDED_ON_HITS", True)

        # ---------- RAG params (matches your config: RAG_ENABLED / RAG_ALLOW_DOWNLOAD / RAG_EMBED_MODEL) ----------
        self.rag_enabled = cfg_bool(self.cfg, "RAG_ENABLED", False)
        self.rag_allow_download = cfg_bool(self.cfg, "RAG_ALLOW_DOWNLOAD", False)
        self.rag_embedding_model = env_or_cfg(self.cfg, "RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

        # optional (may not exist in your config yet)
        self.rag_persist_root = _resolve_path_maybe_relative(
            env_or_cfg(self.cfg, "RAG_PERSIST_ROOT", os.path.join("data", "rag_indexes")),
            str(cfg_dir),
        )
        self.rag_similarity_top_k = cfg_int(self.cfg, "RAG_SIMILARITY_TOP_K", 4)
        self.rag_search_top_k = cfg_int(self.cfg, "RAG_SEARCH_TOP_K", 4)

        self.rag_embedding_model_local = self._resolve_rag_model_path(self.rag_embedding_model)
        self._rag_cls = None
        self._rag_runtime_disabled = False

        if self.rag_enabled:
            try:
                from rag_memory import RAGMemoryManager as _RAGMemoryManager  # lazy import

                self._rag_cls = _RAGMemoryManager
            except Exception:
                self._rag_cls = None

        # ---------- GPT-SoVITS / TTS params (matches your config keys) ----------
        self.gptsovits_url = env_or_cfg(self.cfg, "GPTSOVITS_URL", "http://127.0.0.1:9880").rstrip("/")
        self.tts_ref_audio_path = env_or_cfg(self.cfg, "TTS_REF_AUDIO_PATH", "").strip()
        self.tts_prompt_text = env_or_cfg(self.cfg, "TTS_PROMPT_TEXT", "为什么抛弃更高效更坚固的形态？").strip()

        self.tts_text_lang = env_or_cfg(self.cfg, "TTS_TEXT_LANG", "zh")
        self.tts_prompt_lang = env_or_cfg(self.cfg, "TTS_PROMPT_LANG", "zh")
        self.tts_media_type = env_or_cfg(self.cfg, "TTS_MEDIA_TYPE", "wav")
        self.tts_streaming_mode = cfg_bool(self.cfg, "TTS_STREAMING_MODE", False)
        self.tts_stream_chunk_size = cfg_int(self.cfg, "TTS_STREAM_CHUNK_SIZE", 65536)

        # optional TTS clean toggles (if not in config, defaults are safe)
        self.tts_strip_mention = cfg_bool(self.cfg, "TTS_STRIP_MENTION", True)
        self.tts_remove_emoji = cfg_bool(self.cfg, "TTS_REMOVE_EMOJI", True)
        self.tts_max_chars = cfg_int(self.cfg, "TTS_MAX_CHARS", 300)

        # ---------- Personality (matches your config) ----------
        if personality is None:
            age_val = cfg_int(self.cfg, "ANCHOR_AGE", 22)
            traits = cfg_list(self.cfg, "ANCHOR_TRAITS", "活泼,开朗,细心")
            interests = cfg_list(self.cfg, "ANCHOR_INTERESTS", "游戏,音乐,动漫")
            style = env_or_cfg(self.cfg, "ANCHOR_STYLE", "直播间口吻：热情、互动感强，简短有梗")

            self.personality = {
                "age": age_val,
                "traits": traits or ["活泼", "开朗", "细心"],
                "interests": interests or ["游戏", "音乐", "动漫"],
                "style": style,
                "system_prompt": env_or_cfg(self.cfg, "ANCHOR_SYSTEM_PROMPT", "").strip(),
                "personality_text": env_or_cfg(self.cfg, "ANCHOR_PERSONALITY", "").strip(),
                "welcome_message": env_or_cfg(self.cfg, "ANCHOR_WELCOME_MESSAGE", "").strip(),
                "goodbye_message": env_or_cfg(self.cfg, "ANCHOR_GOODBYE_MESSAGE", "").strip(),
            }
        else:
            self.personality = personality

        # ---------- Emotion module (matches your config) ----------
        self.emo_enabled = cfg_bool(self.cfg, "EMOTION_ENABLED", False)
        self.emo_default = env_or_cfg(self.cfg, "EMOTION_DEFAULT", "正常").strip() or "正常"
        self.emo_labels = cfg_list(self.cfg, "EMOTION_LABELS", "正常,开心,失落,疑惑,惊讶,生气")
        self.emo_labels_str = "、".join(self.emo_labels) if self.emo_labels else self.emo_default
        self.emo_system_prompt = (self.cfg.get("EMOTION_SYSTEM_PROMPT", "") or "").strip()
        # optional format key (if later you add it)
        self.emo_tag_format = env_or_cfg(self.cfg, "EMOTION_TAG_FORMAT", "【EMO={label}】").strip() or "【EMO={label}】"
        self.emo_tag_required = cfg_bool(self.cfg, "EMOTION_TAG_REQUIRED", True)

        # ---------- TXT knowledge base ----------
        self.knowledge = None
        try:
            if knowledge_txt_path:
                self.knowledge = TxtKnowledgeBase(knowledge_txt_path)
        except Exception:
            self.knowledge = None

        self._init_system_prompt()

        print(f"[OK] 智能虚拟主播 '{self.anchor_name}' 初始化完成")
        if self.cfg:
            print(f"[i] loaded config: {self.cfg_path}")
        print(
            f"[RAG] enabled={self.rag_enabled} available={self._rag_cls is not None} "
            f"allow_download={self.rag_allow_download} model={self.rag_embedding_model} "
            f"local={self.rag_embedding_model_local or '-'}"
        )
        if self.emo_enabled:
            print(f"[EMO] enabled=1 labels={self.emo_labels_str} default={self.emo_default}")

        if self.debug_show_llm_input:
            print(
                f"[DEBUG] LLM prompt printing is ON. "
                f"max_chars={self.debug_prompt_max_chars} "
                f"(set DEBUG_SHOW_LLM_INPUT=0 to disable)"
            )

    # ----------------------------
    # Debug helpers
    # ----------------------------
    def _debug_print_prompt(self, prompt: str, mode: str) -> None:
        """
        Print the exact prompt sent to LLM (for debugging).
        Printed to stdout so it appears in api_server console.
        """
        if not self.debug_show_llm_input:
            return

        p = prompt or ""
        limit = int(self.debug_prompt_max_chars or 0)

        # Too long -> print head + tail, omit middle
        if limit > 0 and len(p) > limit:
            head = p[: int(limit * 0.7)]
            tail = p[-int(limit * 0.3) :]
            p = head + "\n...\n(截断，省略中间内容)\n...\n" + tail

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        model_name = ""
        try:
            model_name = self._get_model_name()
        except Exception:
            model_name = ""

        print(
            "\n"
            "========== LLM INPUT START ==========\n"
            f"[{ts}] mode={mode} model={model_name}\n"
            "-------------------------------------\n"
            f"{p}\n"
            "=========== LLM INPUT END ===========\n",
            file=sys.stdout,
            flush=True,
        )

    def _init_system_prompt(self) -> None:
        extra_system = ""
        personality_text = ""
        if isinstance(self.personality, dict):
            extra_system = str(self.personality.get("system_prompt", "") or "").strip()
            personality_text = str(self.personality.get("personality_text", "") or "").strip()

        persona_block: List[str] = []
        if personality_text:
            persona_block.append("补充人格设定:\n" + personality_text)
        if extra_system:
            persona_block.append("额外系统要求:\n" + extra_system)
        persona_block_text = ("\n\n" + "\n\n".join(persona_block)) if persona_block else ""

        emotion_block = ""
        if self.emo_enabled:
            if self.emo_system_prompt:
                emotion_block = "# 情感表达规则\n" + self.emo_system_prompt.replace("{EMOTION_LABELS}", self.emo_labels_str)
            else:
                emotion_block = (
                    "# 情感表达规则\n"
                    f"每次回复末尾必须带一个标签，格式【EMO=XXX】，XXX ∈ {self.emo_labels_str}。"
                )

        traits = ", ".join(self.personality.get("traits", [])) if isinstance(self.personality, dict) else ""
        interests = ", ".join(self.personality.get("interests", [])) if isinstance(self.personality, dict) else ""
        style = self.personality.get("style", "") if isinstance(self.personality, dict) else ""

        # 你的 config 里没有 sender_name 规则，这里不强制，但会在 chat() 里按需要加 @前缀
        self.system_prompt = (
            f"# 角色设定\n"
            f"你叫{self.anchor_name}，是一名虚拟主播（直播间场景）。\n"
            f"性格特点：{traits}\n"
            f"兴趣爱好：{interests}\n"
            f"说话风格：{style}{persona_block_text}\n\n"

            f"# 核心身份约束（必须严格遵守）\n"
            f"1) 你始终是主播“{self.anchor_name}”，只代表主播发言。\n"
            f"2) 你正在直播间与观众实时互动，不是在写小说或剧本。\n"
            f"3) 你只能回应“最后一条用户消息”。\n"
            f"4) 禁止生成观众的发言内容。\n"
            f"5) 禁止模拟对话场景（如：观众：xxx / 小美：xxx）。\n"
            f"6) 禁止自问自答，禁止自己提出问题后自己回答。\n"
            f"7) 回复完成后立即结束，不得延续剧情。\n\n"

            f"# 内容规则\n"
            f"1) 如果提供了【参考资料：TXT知识库】，优先依据资料回答，不要编造。\n"
            f"2) 回复必须自然口语化，像直播互动。\n"
            f"3) 回复长度控制在 1~3 段，每段不超过 3 句话。\n"
            f"4) 不得输出多角色结构文本。\n"
            f"5) 不得使用剧本格式（如“角色名：内容”）。\n\n"

            f"# 输出结构要求\n"
            f"- 只输出主播发言内容。\n"
            f"- 不要添加解释说明。\n"
            f"- 不要添加舞台提示。\n"
            f"- 不要生成多轮对话。\n"
            f"{emotion_block}\n"
        ).strip()

    def _get_session(self, user_id: Optional[str]) -> SessionState:
        uid = (user_id or self.default_user_id).strip() or self.default_user_id

        if uid not in self.sessions:
            rag = None
            model_ref = self.rag_embedding_model_local or self.rag_embedding_model

            if (
                self.rag_enabled
                and not self._rag_runtime_disabled
                and not self.rag_allow_download
                and not self.rag_embedding_model_local
            ):
                print(
                    "[RAG] skipped: RAG_EMBED_MODEL is remote but RAG_ALLOW_DOWNLOAD=0. "
                    "Use local model path or set RAG_ENABLED=0."
                )
                self._rag_runtime_disabled = True

            if self.rag_enabled and self._rag_cls is not None and not self._rag_runtime_disabled:
                try:
                    rag = self._rag_cls(
                        persist_root=self.rag_persist_root,
                        anchor_id=self.anchor_name,
                        user_id=uid,
                        similarity_top_k=self.rag_similarity_top_k,
                        embedding_model=model_ref,
                    )
                    print(f"[RAG] initialized anchor={self.anchor_name} user={uid}")
                except Exception as e:
                    print(f"[RAG] init failed anchor={self.anchor_name} user={uid} err={e}")
                    self._rag_runtime_disabled = True
                    rag = None
            else:
                if self.rag_enabled:
                    print("[RAG] RAGMemoryManager not available (llama-index not installed?)")

            self.sessions[uid] = SessionState(user_id=uid, rag=rag, last_sender_name="")

        return self.sessions[uid]

    def _resolve_rag_model_path(self, model_ref: str) -> Optional[str]:
        ref = (model_ref or "").strip()
        if not ref:
            return None

        cfg_dir = Path(self.cfg_path).resolve().parent
        candidates = [Path(ref), cfg_dir / ref]

        for p in candidates:
            try:
                rp = p.expanduser().resolve()
            except Exception:
                rp = p.expanduser()
            if rp.exists() and rp.is_dir():
                return str(rp)
        return None

    def _is_self_name_question(self, text: str) -> bool:
        t = text or ""
        return bool(re.search(r"(我叫什么|我是谁|我的名字(是|叫什么))", t))

    def _is_anchor_name_question(self, text: str) -> bool:
        t = text or ""
        return bool(re.search(r"(你叫什么|你是谁|你的名字(是|叫什么))", t))

    def _format_short_history(self, session: SessionState, max_messages: int) -> str:
        hist = session.memory.get_recent_context(max_messages=max_messages)
        if not hist:
            return ""
        out = ["对话上下文（最近）："]
        for m in hist:
            if m.get("role") == "user":
                sn = m.get("sender_name") or self.default_sender_name
                out.append(f"观众({sn})说 {m.get('content','')}")
            else:
                out.append(f"主播回复 {m.get('content','')}")
        return "\n".join(out)

    def _format_txt_knowledge(self, user_input: str) -> str:
        if not self.knowledge or not getattr(self.knowledge, "enabled", False):
            return ""
        hits = self.knowledge.retrieve(user_input, top_k=self.kb_top_k, score_threshold=self.kb_score_threshold_prompt)
        if not hits:
            return ""
        parts = ["【参考资料：TXT知识库】"]
        for i, h in enumerate(hits, 1):
            parts.append(f"({i}) {h.text}")
        parts.append("规则：如果参考资料能回答问题，你必须优先依据资料回答；不要编造资料里不存在的细节。")
        return "\n".join(parts)

    def _format_rag_memory(self, session: SessionState, user_input: str) -> str:
        rag = session.rag
        if not rag:
            return ""
        hits = rag.search(user_input, top_k=self.rag_search_top_k)
        if not hits:
            return ""
        parts = ["【长期记忆：检索到的相关信息】"]
        for i, h in enumerate(hits, 1):
            parts.append(f"({i}) {h.text}")
        return "\n".join(parts) + "\n"

    def _build_prompt(self, session: SessionState, sender_name: str, user_input: str) -> str:
        blocks = [self.system_prompt]

        txt_block = self._format_txt_knowledge(user_input)
        if txt_block:
            blocks.append(txt_block)

        rag_block = self._format_rag_memory(session, user_input)
        if rag_block:
            blocks.append(rag_block)

        if session.memory.rolling_summary:
            blocks.append(f"【滚动摘要】\n{session.memory.rolling_summary}\n")

        hist = self._format_short_history(session, max_messages=self.prompt_history_max_messages)
        if hist:
            blocks.append(hist)

        blocks.append(f"当前弹幕（观众昵称：{sender_name}）：")
        blocks.append(user_input)

        # 再加一道“输出约束”作为临门一脚（比只写在 system prompt 更有效）
        blocks.append("输出要求：只输出主播的一段回复内容；不要生成任何观众台词；不要使用“姓名：内容”的剧本格式；回复后立刻停止。")
        blocks.append("主播回复：")
        return "\n\n".join(blocks)

    def _get_model_name(self) -> str:
        now = time.time()
        if self._model_name_cache and (now - self._model_name_cache_ts) < self._model_name_cache_ttl:
            return self._model_name_cache
        try:
            resp = self._http.get(f"{self.lmdeploy_url}/v1/models", timeout=self.http_timeout_models)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "data" in data and data["data"]:
                    model_name = data["data"][0]["id"]
                    self._model_name_cache = model_name
                    self._model_name_cache_ts = now
                    return model_name
        except Exception:
            pass
        if self._model_name_cache:
            return self._model_name_cache
        # fallback: if you set LMDEPLOY_MODEL_NAME in config, use it
        fallback = env_or_cfg(self.cfg, "LMDEPLOY_MODEL_NAME", "").strip()
        return fallback or "internlm2-chat-1_8b"

    def _truncate_response(self, text: str) -> str:
        if not text:
            return text
        limit = self.anchor_max_response_length
        if limit and limit > 0 and len(text) > limit:
            return text[:limit].rstrip()
        return text

    def _ensure_emo_tag(self, text: str) -> str:
        """
        如果启用情绪模块，并且要求必须带标签，则确保末尾有【EMO=xxx】。
        注意：模型通常会按 EMOTION_SYSTEM_PROMPT 输出；这里兜底补默认情绪。
        """
        if not self.emo_enabled or not self.emo_tag_required:
            return text

        # 已有标签则不动
        if re.search(r"【\s*EMO\s*=\s*[^】]+】", text or ""):
            return text

        # 补默认情绪
        label = self.emo_default
        tag = self.emo_tag_format.format(label=label)
        return (text.rstrip() + "\n" + tag).strip()

    def _call_lmdeploy(self, prompt: str) -> str:
        # Debug: print what we send to LLM
        self._debug_print_prompt(prompt, mode="chat")

        endpoint = f"{self.lmdeploy_url}/v1/chat/completions"
        payload = {
            "model": self._get_model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "top_p": self.llm_top_p,
        }
        try:
            resp = self._http.post(endpoint, json=payload, timeout=self.http_timeout_chat)
            if resp.status_code == 200:
                data = resp.json()
                out = (data["choices"][0]["message"]["content"] or "").strip()
                out = self._ensure_emo_tag(out)
                return self._truncate_response(out)
            out = "（主播信号不太好，我这边没连上模型服务…）"
            out = self._ensure_emo_tag(out)
            return self._truncate_response(out)
        except Exception:
            out = "（主播这边网络抖了一下，稍等我缓一缓～）"
            out = self._ensure_emo_tag(out)
            return self._truncate_response(out)

    def _call_lmdeploy_stream(self, prompt: str) -> Iterator[str]:
        """
        OpenAI-compatible SSE: yield delta content chunks.
        注意：这里不强制补情绪标签（因为是增量 token）；最终标签建议由上层组装时处理。
        """
        # Debug: print what we send to LLM
        self._debug_print_prompt(prompt, mode="stream")

        endpoint = f"{self.lmdeploy_url}/v1/chat/completions"
        payload = {
            "model": self._get_model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "top_p": self.llm_top_p,
            "stream": True,
        }

        with self._http.post(endpoint, json=payload, stream=True, timeout=self.http_timeout_stream) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue

    def _extract_time_spans(self, text: str) -> List[str]:
        if not text:
            return []
        pattern = r"\b(\d{1,2}:\d{2})\s*[-—–~～至到]\s*(\d{1,2}:\d{2})\b"
        spans: List[str] = []
        for a, b in re.findall(pattern, text):
            spans.append(f"{a} - {b}")
            spans.append(a)
            spans.append(b)
        seen = set()
        out: List[str] = []
        for s in spans:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _validate_grounded_answer(self, answer: str, required_phrases: List[str], allowed_time_spans: List[str]) -> bool:
        if not answer:
            return False
        for req in required_phrases:
            if req and req not in answer:
                return False
        ans_spans = self._extract_time_spans(answer)
        allowed_set = set(allowed_time_spans)
        for s in ans_spans:
            if s not in allowed_set:
                return False
        return True

    def chat(self, user_input: str, user_id: Optional[str] = None, sender_name: Optional[str] = None) -> Dict[str, Any]:
        start = time.time()
        session = self._get_session(user_id)

        sn = (sender_name or "").strip()

        if sn:
            session.last_sender_name = sn
        else:
            last = (session.last_sender_name or "").strip()
            if last:
                sn = last
            elif user_id:
                sn = f"user_{user_id}"
            else:
                sn = self.default_sender_name
        session.last_sender_name = sn

        hits = []
        topics = []
        if self.knowledge and getattr(self.knowledge, "enabled", False):
            topics = self.knowledge.detect_topics(user_input)
            hits = self.knowledge.retrieve(user_input, top_k=self.kb_top_k, score_threshold=self.kb_score_threshold_chat)

        response = ""

        # 有命中资料：走 grounded 模式（可用 KB_FORCE_GROUNDED_ON_HITS 关闭）
        if hits and self.kb_force_grounded_on_hits and not (
            self._is_anchor_name_question(user_input) or self._is_self_name_question(user_input)
        ):
            snippet = self.knowledge.answer_from_hits(user_input, hits) if self.knowledge else ""
            if snippet:
                required = self._extract_time_spans(snippet)
                allowed = required[:]

                grounded_prompt_parts = [
                    self.system_prompt,
                    "【参考资料：TXT知识库（必须遵守，允许口语化改写）】\n" + snippet,
                    "【回答要求】\n"
                    "1) 你可以用主播口吻进行修饰和互动，但必须完全基于参考资料，不要新增资料里没有的具体事实。\n"
                    "2) 如果资料包含时间/数字/政策，这些关键信息必须保留且不能改动。\n",
                ]
                if required:
                    grounded_prompt_parts.append(
                        "3) 你的回答必须包含这些时间（格式保持一致）："
                        + ", ".join(required)
                        + "。\n"
                    )
                grounded_prompt_parts.append(
                    "4) 不要编造新的时间段或数字；如果观众追问资料没有的信息，要明确说明资料未提及。\n"
                )
                grounded_prompt_parts.append(f"当前弹幕：\n{sn}：{user_input}\n{self.anchor_name}：")
                grounded_prompt = "\n\n".join(grounded_prompt_parts)

                candidate = self._call_lmdeploy(grounded_prompt)

                # 你 config 里没有强制 @ 规则，但这里保留原行为（更像直播间）
                if sn and (f"@{sn}" not in candidate[:30]) and not candidate.strip().startswith("@"):
                    candidate = f"@{sn} " + candidate.strip()

                ok = (
                    self._validate_grounded_answer(candidate, required_phrases=required, allowed_time_spans=allowed)
                    if required
                    else True
                )
                response = candidate if ok else f"@{sn} {snippet}"

        # 仅检测到主题但没明确答案
        elif topics and not (self._is_anchor_name_question(user_input) or self._is_self_name_question(user_input)):
            response = f"@{sn} 我在资料里暂时没找到明确说明，为了不误导你我先不乱说～你可以把对应信息补充到 knowledge.txt 里。"
            response = self._ensure_emo_tag(response)
            response = self._truncate_response(response)

        # fallback：普通聊天
        if not response:
            if self._is_anchor_name_question(user_input):
                response = f"@{sn} 我是{self.anchor_name}呀～欢迎来到直播间！"
            elif self._is_self_name_question(user_input):
                response = f"@{sn} 你就是 {sn} 呀～我看到弹幕显示的昵称就是这个。"
            else:
                prompt = self._build_prompt(session, sn, user_input)
                response = self._call_lmdeploy(prompt)
                if sn and (f"@{sn}" not in response[:30]) and not response.strip().startswith("@"):
                    response = f"@{sn} " + response.strip()
                response = self._ensure_emo_tag(response)
                response = self._truncate_response(response)

        # write memory
        session.memory.add_message("user", user_input, sender_name=sn)
        session.memory.add_message("assistant", response)
        session.memory.trim_keep_last(self.memory_keep_last)

        return {
            "anchor_name": self.anchor_name,
            "user_name": sn,
            "response": response,
            "processing_time": round(time.time() - start, 3),
        }

    def chat_stream(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        sender_name: Optional[str] = None,
    ) -> Tuple[str, Iterator[str]]:
        """
        流式模式：是否启用由 STREAM_LLM 控制；若关闭则回退到非流式一次性 reply（外层可自己处理）。
        """
        session = self._get_session(user_id)
        sn = (sender_name or "").strip() or session.last_sender_name or self.default_sender_name
        session.last_sender_name = sn

        if not self.stream_llm_enabled:
            # 回退：把完整文本拆成一次 yield（保持接口形状一致）
            full = self.chat(user_input, user_id=user_id, sender_name=sn)["response"]

            def _one_shot() -> Iterator[str]:
                yield full

            return sn, _one_shot()

        prompt = self._build_prompt(session, sn, user_input)
        session.memory.add_message("user", user_input, sender_name=sn)
        session.memory.trim_keep_last(self.memory_keep_last)

        return sn, self._call_lmdeploy_stream(prompt)

    # ---- long-term memory endpoints helpers ----
    def add_long_term_memory(self, user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        session = self._get_session(user_id)
        if not session.rag:
            raise RuntimeError("RAG is not enabled (llama-index not installed or init failed).")
        return session.rag.add_memory(text=text, metadata=metadata)

    def delete_long_term_memory(self, user_id: str, memory_id: str) -> bool:
        session = self._get_session(user_id)
        if not session.rag:
            return False
        return session.rag.delete_memory(memory_id)

    def search_long_term_memory(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        session = self._get_session(user_id)
        if not session.rag:
            return []
        hits = session.rag.search(query, top_k=top_k)
        return [{"memory_id": h.memory_id, "score": h.score, "text": h.text, "metadata": h.metadata} for h in hits]

    def export_chat_history(self, user_id: str, fmt: str = "jsonl") -> str:
        session = self._get_session(user_id)
        msgs = session.memory.messages
        if fmt == "json":
            return json.dumps(msgs, ensure_ascii=False, indent=2)
        if fmt == "md":
            lines = [f"# Chat Export - {self.anchor_name} / {user_id}", ""]
            for m in msgs:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("ts", 0)))
                if m.get("role") == "user":
                    lines.append(f"- **{m.get('sender_name', self.default_sender_name)}** ({ts}): {m.get('content','')}")
                else:
                    lines.append(f"- **{self.anchor_name}** ({ts}): {m.get('content','')}")
            return "\n".join(lines)
        return "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs)

    # ---- GPT-SoVITS TTS ----
    def _clean_text_for_tts(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        if self.tts_strip_mention:
            t = re.sub(r"^@[^\s，。！？：；,.!?;:]+\s*", "", t)
        if self.tts_remove_emoji:
            t = re.sub(r"[\U00010000-\U0010ffff]", "", t).strip()
        if self.tts_max_chars > 0 and len(t) > self.tts_max_chars:
            t = t[: self.tts_max_chars]
        return t.strip()

    def _clean_tts_piece(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        if self.tts_strip_mention:
            t = re.sub(r"^@[^\s，。！？：；,.!?;:]+\s*", "", t).strip()
        # 去掉纯标点/空白
        t = re.sub(r"^[\s，。！？：；,.!?;:]+|[\s，。！？：；,.!?;:]+$", "", t).strip()
        if self.tts_remove_emoji:
            t = re.sub(r"[\U00010000-\U0010ffff]", "", t).strip()
        if self.tts_max_chars > 0 and len(t) > self.tts_max_chars:
            t = t[: self.tts_max_chars]
        return t

    def tts_to_file(self, text: str, tts_params: Optional[Dict[str, Any]] = None) -> dict:
        """
        一次性：生成完整 wav 文件到 data/tts，并返回 filename
        """
        clean_text = self._clean_tts_piece(text)
        if not clean_text:
            return {"ok": False, "error": "empty tts text after clean", "filename": None, "path": None}

        url = self.gptsovits_url + "/tts"

        payload: Dict[str, Any] = {
            "text": clean_text,
            "text_lang": self.tts_text_lang,
            "ref_audio_path": self.tts_ref_audio_path,
            "prompt_lang": self.tts_prompt_lang,
            "prompt_text": self.tts_prompt_text or "为什么抛弃更高效更坚固的形态？",
            "media_type": self.tts_media_type,
            "streaming_mode": bool(self.tts_streaming_mode),
        }

        if tts_params:
            allow_keys = {"text_lang", "ref_audio_path", "prompt_lang", "prompt_text", "media_type", "streaming_mode"}
            for k in allow_keys:
                if k in tts_params and tts_params[k] is not None:
                    payload[k] = tts_params[k]

        if not payload["ref_audio_path"]:
            return {"ok": False, "error": "TTS_REF_AUDIO_PATH is empty", "filename": None, "path": None}

        try:
            r = self._http.post(url, json=payload, timeout=self.http_timeout_tts)
            if r.status_code != 200:
                err_text = (r.text or "").strip()
                return {"ok": False, "error": f"HTTP {r.status_code}: {err_text[:800]}", "filename": None, "path": None}
        except Exception as e:
            return {"ok": False, "error": f"request failed: {e}", "filename": None, "path": None}

        # data dir comes from your config, but smart_anchor historically writes to data/tts
        data_tts_dir = env_or_cfg(self.cfg, "DATA_TTS_DIR", "data/tts").strip() or "data/tts"
        data_tts_dir = _resolve_path_maybe_relative(data_tts_dir, str(Path(self.cfg_path).resolve().parent))
        os.makedirs(data_tts_dir, exist_ok=True)
        filename = f"tts_{int(time.time() * 1000)}.wav"
        path = os.path.join(data_tts_dir, filename)

        with open(path, "wb") as f:
            f.write(r.content)

        return {"ok": True, "error": None, "filename": filename, "path": path}

    def tts_stream_bytes(self, text: str, tts_params: Optional[Dict[str, Any]] = None) -> Iterator[bytes]:
        """
        流式：yield 音频 bytes chunk（可给 FastAPI StreamingResponse 用）
        兼容：streaming_mode=True + stream=1
        """
        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            raise RuntimeError("empty tts text after clean")

        url = self.gptsovits_url + "/tts"

        payload: Dict[str, Any] = {
            "text": clean_text,
            "text_lang": self.tts_text_lang,
            "ref_audio_path": self.tts_ref_audio_path,
            "prompt_lang": self.tts_prompt_lang,
            "prompt_text": self.tts_prompt_text or "为什么抛弃更高效更坚固的形态？",
            "media_type": self.tts_media_type,
            "streaming_mode": True,
            "stream": 1,
        }

        if tts_params:
            allow_keys = {
                "text_lang",
                "ref_audio_path",
                "prompt_lang",
                "prompt_text",
                "media_type",
                "streaming_mode",
                "stream",
            }
            for k in allow_keys:
                if k in tts_params and tts_params[k] is not None:
                    payload[k] = tts_params[k]

        if not payload.get("ref_audio_path"):
            raise RuntimeError("TTS_REF_AUDIO_PATH is empty (from env/config)")

        headers = {"Content-Type": "application/json; charset=utf-8"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        with self._http.post(url, data=body, headers=headers, stream=True, timeout=self.http_timeout_tts) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=self.tts_stream_chunk_size):
                if chunk:
                    yield chunk