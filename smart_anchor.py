# smart_anchor.py
from __future__ import annotations

import os
import time
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple

import requests
from txt_knowledge import TxtKnowledgeBase

try:
    from rag_memory import RAGMemoryManager
except Exception:
    RAGMemoryManager = None  # type: ignore


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


def cfg_bool(cfg: Dict[str, str], key: str, default: bool = False) -> bool:
    v = (cfg.get(key, "") or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _split_csv(s: str) -> List[str]:
    if not s:
        return []
    # allow Chinese comma/semicolon
    s = s.replace("，", ",").replace("；", ";")
    parts: List[str] = []
    for chunk in re.split(r"[,;]", s):
        t = chunk.strip()
        if t:
            parts.append(t)
    return parts


def resolve_config_path() -> str:
    """
    Priority:
    1) env CONFIG_FILE
    2) project root "config.txt" (same dir as this file's parent)
    """
    env_path = os.environ.get("CONFIG_FILE", "").strip()
    if env_path:
        return env_path

    here = Path(__file__).resolve().parent
    default_path = here / "config.txt"
    return str(default_path)


def env_or_cfg(cfg: Dict[str, str], key: str, default: str = "") -> str:
    """
    Prefer environment variable (useful when start_all.bat sets them),
    fallback to config.txt, then default.
    """
    v = os.environ.get(key, "").strip()
    if v:
        return v
    return cfg.get(key, default).strip()


@dataclass
class ConversationMemory:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    rolling_summary: str = ""

    def add_message(self, role: str, content: str, sender_name: Optional[str] = None) -> None:
        self.messages.append(
            {"role": role, "content": content, "sender_name": sender_name, "ts": time.time()}
        )
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def get_recent_context(self, max_messages: int = 6) -> List[Dict[str, Any]]:
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

        # Basic endpoints / paths (env > config > args/default)
        lmdeploy_url = env_or_cfg(self.cfg, "LMDEPLOY_URL", lmdeploy_url)
        name = env_or_cfg(self.cfg, "ANCHOR_NAME", name)

        # knowledge txt (env/config override)
        knowledge_txt_path = env_or_cfg(self.cfg, "KNOWLEDGE_TXT", knowledge_txt_path)
        # if relative path, resolve relative to config file directory
        try:
            kp = Path(knowledge_txt_path)
            if not kp.is_absolute():
                knowledge_txt_path = str(Path(cfg_path).resolve().parent / kp)
        except Exception:
            pass

        self.lmdeploy_url = lmdeploy_url.rstrip("/")
        self.anchor_name = name
        self.sessions: Dict[str, SessionState] = {}
        self.default_user_id = "default_room"

        # ===== LLM generation params (env > config > default) =====
        def _safe_float(x: str, default: float) -> float:
            try:
                return float(str(x).strip())
            except Exception:
                return default

        def _safe_int(x: str, default: int) -> int:
            try:
                return int(str(x).strip())
            except Exception:
                return default

        self.llm_temperature = _safe_float(env_or_cfg(self.cfg, "ANCHOR_TEMPERATURE", "0.7"), 0.7)
        self.llm_top_p = _safe_float(env_or_cfg(self.cfg, "ANCHOR_TOP_P", "0.9"), 0.9)
        # New key (recommended): ANCHOR_MAX_TOKENS. Default is 300 for compatibility.
        self.llm_max_tokens = _safe_int(env_or_cfg(self.cfg, "ANCHOR_MAX_TOKENS", "300"), 300)

        # ===== GPT-SoVITS / TTS params (env > config > default) =====
        self.gptsovits_url = env_or_cfg(self.cfg, "GPTSOVITS_URL", "http://127.0.0.1:9880").rstrip("/")
        self.tts_ref_audio_path = env_or_cfg(self.cfg, "TTS_REF_AUDIO_PATH", "").strip()
        self.tts_prompt_text = env_or_cfg(self.cfg, "TTS_PROMPT_TEXT", "为什么抛弃更高效更坚固的形态？").strip()

        # TTS generation params (env > config > default)
        self.tts_text_lang = env_or_cfg(self.cfg, "TTS_TEXT_LANG", "zh")
        self.tts_prompt_lang = env_or_cfg(self.cfg, "TTS_PROMPT_LANG", "zh")
        self.tts_media_type = env_or_cfg(self.cfg, "TTS_MEDIA_TYPE", "wav")
        self.tts_streaming_mode = cfg_bool(self.cfg, "TTS_STREAMING_MODE", False)
        try:
            self.tts_stream_chunk_size = int(env_or_cfg(self.cfg, "TTS_STREAM_CHUNK_SIZE", "65536"))
        except Exception:
            self.tts_stream_chunk_size = 65536


        # Personality: from config unless explicitly passed in personality arg
        if personality is None:
            age_str = env_or_cfg(self.cfg, "ANCHOR_AGE", "22")
            try:
                age_val = int(age_str)
            except Exception:
                age_val = 22

            traits = _split_csv(env_or_cfg(self.cfg, "ANCHOR_TRAITS", "活泼,开朗,细心"))
            interests = _split_csv(env_or_cfg(self.cfg, "ANCHOR_INTERESTS", "游戏,音乐,动漫"))
            style = env_or_cfg(self.cfg, "ANCHOR_STYLE", "直播间口吻：热情、会@观众、互动感强，简短有梗")

            self.personality = {
                "age": age_val,
                "traits": traits or ["活泼", "开朗", "细心"],
                "interests": interests or ["游戏", "音乐", "动漫"],
                "style": style,
                "system_prompt": env_or_cfg(self.cfg, "ANCHOR_SYSTEM_PROMPT", "").strip(),
                "personality_text": env_or_cfg(self.cfg, "ANCHOR_PERSONALITY", "").strip(),
            }
        else:
            self.personality = personality

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

# 在 smart_anchor.py 的 _init_system_prompt 方法中，修改系统提示词部分：

    def _init_system_prompt(self) -> None:
        extra_system = ""
        personality_text = ""
        if isinstance(self.personality, dict):
            extra_system = str(self.personality.get("system_prompt", "") or "").strip()
            personality_text = str(self.personality.get("personality_text", "") or "").strip()

        persona_block = []
        if personality_text:
            persona_block.append("补充人格设定：\n" + personality_text)
        if extra_system:
            persona_block.append("额外系统要求：\n" + extra_system)
        persona_block_text = ("\n\n" + "\n\n".join(persona_block)) if persona_block else ""

        # 检查是否启用了情感模块
        emo_enabled = cfg_bool(self.cfg, "EMOTION_ENABLED", False)
        emotion_block = ""
        
        if emo_enabled:
            emotion_labels = _split_csv(self.cfg.get("EMOTION_LABELS", "正常,开心,失落,疑惑,惊讶,生气"))
            emotion_labels_str = "、".join(emotion_labels)
            
            # 从配置中读取情感系统提示词，如果没有则使用默认
            emotion_system_prompt = self.cfg.get("EMOTION_SYSTEM_PROMPT", "").strip()
            if emotion_system_prompt:
                # 替换占位符
                emotion_system_prompt = emotion_system_prompt.replace("{EMOTION_LABELS}", emotion_labels_str)
                emotion_block = f"""
# 情感表达规则
{emotion_system_prompt}
""".strip()
            else:
                # 默认的情感提示词
                emotion_block = f"""
# 情感表达规则
你是一个有情感的主播，请在回复中适当表达情感。
你必须在每次回复的末尾添加一个情感标签，格式为【emo=XXX】，XXX必须是以下情感之一：{emotion_labels_str}。
情感标签的示例用法：
- "今天真开心见到大家！【emo=开心】"
- "这个问题有点复杂...【emo=疑惑】"
- "什么？！真的吗？【emo=惊讶】"
- "不要这样说嘛...【emo=失落】"
- "我有点生气了！【emo=生气】"
- "好的，我们继续直播吧。【emo=正常】"

规则：
1. 每次回复必须有且只有一个情感标签，放在回复的末尾
2. 根据你的回复内容和语境选择最合适的情感
3. 情感标签必须用中文方括号包裹，格式为【emo=XXX】
4. 情感标签不影响你回复的正文内容，它只是表示你的情感状态
""".strip()

        self.system_prompt = f"""# 角色设定
你叫{self.anchor_name}，是一名虚拟主播（直播间场景）。
性格特点：{', '.join(self.personality['traits'])}
兴趣爱好：{', '.join(self.personality['interests'])}
说话风格：{self.personality['style']}{persona_block_text}

# 直播间规则（重要）
1) 你始终是主播{self.anchor_name}，不要把自己当成观众。
2) 每条弹幕都有发送者名字 sender_name，你回答时优先@对方（例如"@小明 …"）。
3) 如果提供了【参考资料：TXT知识库】，你必须优先依据资料回答；资料没有的内容不要瞎编。
4) 回复要自然有互动，适当抛话题带节奏；尽量别太长（1~3 段）。
{emotion_block}
""".strip()
        
        # 调试输出
        if emo_enabled:
            print(f"[EMO] 情感模块已启用，情感标签: {emotion_labels_str}")
            print(f"[EMO] 情感系统提示词长度: {len(emotion_block)} 字符")
    def _get_session(self, user_id: Optional[str]) -> SessionState:
        uid = (user_id or self.default_user_id).strip() or self.default_user_id

        if uid not in self.sessions:
            rag = None
            if RAGMemoryManager is not None:
                try:
                    rag = RAGMemoryManager(
                        persist_root=os.path.join("data", "rag_indexes"),
                        anchor_id=self.anchor_name,
                        user_id=uid,
                        similarity_top_k=4,
                    )
                    print(f"[RAG] initialized ✓ anchor={self.anchor_name} user={uid}")
                except Exception as e:
                    print(f"[RAG] init failed ✗ anchor={self.anchor_name} user={uid} err={e}")
                    rag = None
            else:
                print("[RAG] RAGMemoryManager not available (llama-index not installed?)")

            self.sessions[uid] = SessionState(user_id=uid, rag=rag)

        return self.sessions[uid]

    def _is_self_name_question(self, text: str) -> bool:
        return bool(re.search(r"我叫什么|我是谁|我的名字是什么", text or ""))

    def _is_anchor_name_question(self, text: str) -> bool:
        return bool(re.search(r"你叫什么|你是谁|你的名字是什么", text or ""))

    def _format_short_history(self, session: SessionState, max_messages: int = 6) -> str:
        hist = session.memory.get_recent_context(max_messages=max_messages)
        if not hist:
            return ""
        out = ["对话上下文（最近）："]
        for m in hist:
            if m["role"] == "user":
                sn = m.get("sender_name") or "观众"
                out.append(f"{sn}：{m['content']}")
            else:
                out.append(f"{self.anchor_name}：{m['content']}")
        return "\n".join(out)

    def _format_txt_knowledge(self, user_input: str) -> str:
        if not self.knowledge or not getattr(self.knowledge, "enabled", False):
            return ""
        hits = self.knowledge.retrieve(user_input, top_k=4, score_threshold=0.35)
        if not hits:
            return ""
        parts = ["【参考资料：TXT知识库】"]
        for i, h in enumerate(hits, 1):
            parts.append(f"({i}) {h.text}")
        parts.append("规则：如果参考资料能回答问题，你必须优先依据资料回答；不要编造资料里不存在的细节。\n")
        return "\n".join(parts)

    def _format_rag_memory(self, session: SessionState, user_input: str) -> str:
        rag = session.rag
        if not rag:
            return ""
        hits = rag.search(user_input, top_k=4)
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
            blocks.append(f"【滚动摘要】{session.memory.rolling_summary}\n")
        hist = self._format_short_history(session, max_messages=6)
        if hist:
            blocks.append(hist)

        blocks.append("当前弹幕：")
        blocks.append(f"{sender_name}：{user_input}")
        blocks.append(f"{self.anchor_name}：")
        return "\n\n".join(blocks)

    def _get_model_name(self) -> str:
        try:
            resp = requests.get(f"{self.lmdeploy_url}/v1/models", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "data" in data and data["data"]:
                    return data["data"][0]["id"]
        except Exception:
            pass
        return "internlm2-chat-1_8b"

    def _call_lmdeploy(self, prompt: str) -> str:
        endpoint = f"{self.lmdeploy_url}/v1/chat/completions"
        payload = {
            "model": self._get_model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "top_p": self.llm_top_p,
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            return "（主播信号不太好，我这边没连上模型服务…）"
        except Exception:
            return "（主播这边网络抖了一下，稍等我缓一缓～）"

    # ✅ 新增：LMDeploy 流式（SSE）
    def _call_lmdeploy_stream(self, prompt: str) -> Iterator[str]:
        """
        OpenAI兼容 SSE：逐段 yield delta content
        """
        endpoint = f"{self.lmdeploy_url}/v1/chat/completions"
        payload = {
            "model": self._get_model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "top_p": self.llm_top_p,
            "stream": True,
        }

        with requests.post(endpoint, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
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
        pattern = r"\b(\d{1,2}:\d{2})\s*[-–—~～至到]\s*(\d{1,2}:\d{2})\b"
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

    def _validate_grounded_answer(
        self, answer: str, required_phrases: List[str], allowed_time_spans: List[str]
    ) -> bool:
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

    def chat(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        sender_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        session = self._get_session(user_id)

        sn = (sender_name or "").strip() or session.last_sender_name or "观众"
        session.last_sender_name = sn

        hits = []
        topics = []
        if self.knowledge and getattr(self.knowledge, "enabled", False):
            topics = self.knowledge.detect_topics(user_input)
            hits = self.knowledge.retrieve(user_input, top_k=4, score_threshold=0.18)

        response = ""
        if hits and not (self._is_anchor_name_question(user_input) or self._is_self_name_question(user_input)):
            snippet = self.knowledge.answer_from_hits(user_input, hits) if self.knowledge else ""
            if snippet:
                required = self._extract_time_spans(snippet)
                allowed = required[:]

                grounded_prompt = "\n\n".join(
                    [
                        self.system_prompt,
                        "【参考资料：TXT知识库（必须遵守，允许口语化改写）】\n" + snippet,
                        "【回答要求】\n"
                        "1) 你可以用主播口吻进行修饰和互动，但必须完全基于参考资料，不要新增资料里没有的具体事实。\n"
                        "2) 如果资料包含时间/数字/政策，这些关键事实必须保留且不能改动。\n"
                        + (
                            "3) 你的回答必须包含这些时间（格式保持一致）："
                            + ", ".join(required)
                            + "。\n"
                            if required
                            else ""
                        )
                        + "4) 不要编造新的时间段或数字；如果观众追问资料没有的信息，要明确说明资料未提及。\n",
                        "当前弹幕：\n" + f"{sn}：{user_input}\n{self.anchor_name}：",
                    ]
                )

                candidate = self._call_lmdeploy(grounded_prompt)
                if sn and (f"@{sn}" not in candidate[:30]) and not candidate.strip().startswith("@"):
                    candidate = f"@{sn} " + candidate.strip()

                ok = (
                    self._validate_grounded_answer(candidate, required_phrases=required, allowed_time_spans=allowed)
                    if required
                    else True
                )
                response = candidate if ok else f"@{sn} {snippet}"
        elif topics and not (self._is_anchor_name_question(user_input) or self._is_self_name_question(user_input)):
            response = f"@{sn} 我在资料里暂时没找到明确说明，为了不误导你我先不乱说～你可以把对应信息补充到 knowledge.txt 里。"

        if response:
            pass
        elif self._is_anchor_name_question(user_input):
            response = f"@{sn} 我是{self.anchor_name}呀～欢迎来直播间！"
        elif self._is_self_name_question(user_input):
            response = f"@{sn} 你就是 {sn}～我这边看到弹幕昵称就是这个！"
        else:
            prompt = self._build_prompt(session, sn, user_input)
            response = self._call_lmdeploy(prompt)
            if sn and (f"@{sn}" not in response[:30]) and not response.strip().startswith("@"):
                response = f"@{sn} " + response.strip()

        session.memory.add_message("user", user_input, sender_name=sn)
        session.memory.add_message("assistant", response)

        return {
            "anchor_name": self.anchor_name,
            "user_name": sn,
            "response": response,
            "processing_time": round(time.time() - start, 3),
        }

    # ✅ 新增：流式聊天（返回 sender_name + token iterator）
    def chat_stream(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        sender_name: Optional[str] = None,
    ) -> Tuple[str, Iterator[str]]:
        """
        注意：流式模式不做 grounded 回退（因为流式无法"回滚"）。
        仍然会把 TXT/RAG/历史拼进 prompt。
        """
        session = self._get_session(user_id)

        sn = (sender_name or "").strip() or session.last_sender_name or "观众"
        session.last_sender_name = sn

        prompt = self._build_prompt(session, sn, user_input)

        # 先写入 user 消息（assistant 最终在 API 层写入）
        session.memory.add_message("user", user_input, sender_name=sn)

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
        return [
            {"memory_id": h.memory_id, "score": h.score, "text": h.text, "metadata": h.metadata}
            for h in hits
        ]

    def export_chat_history(self, user_id: str, fmt: str = "jsonl") -> str:
        session = self._get_session(user_id)
        msgs = session.memory.messages
        if fmt == "json":
            return json.dumps(msgs, ensure_ascii=False, indent=2)
        if fmt == "md":
            lines = [f"# Chat Export - {self.anchor_name} / {user_id}", ""]
            for m in msgs:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("ts", 0)))
                if m["role"] == "user":
                    lines.append(f"- **{m.get('sender_name','观众')}** ({ts}): {m['content']}")
                else:
                    lines.append(f"- **{self.anchor_name}** ({ts}): {m['content']}")
            return "\n".join(lines)
        return "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs)

    # ---- GPT-SoVITS TTS ----
    def _clean_text_for_tts(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        t = re.sub(r"^@[^\s，,。:：!?！？]+\s*", "", t)
        return t.strip()

    def tts_to_file(self, text: str, tts_params: Optional[Dict[str, Any]] = None) -> dict:
        """
        一次性：生成完整 wav 文件到 data/tts，并返回 filename。
        """
        clean_text = self._clean_tts_piece(text)
        if not clean_text:
            return {"ok": False, "error": "empty tts text after clean", "filename": None, "path": None}

        url = self.gptsovits_url + "/tts"

        payload = {
            "text": clean_text,
            "text_lang": self.tts_text_lang,
            "ref_audio_path": self.tts_ref_audio_path,
            "prompt_lang": self.tts_prompt_lang,
            "prompt_text": self.tts_prompt_text or "为什么抛弃更高效更坚固的形态？",
            "media_type": self.tts_media_type,
            "streaming_mode": bool(self.tts_streaming_mode),
        }

        if tts_params:
            allow_keys = {
                "text_lang",
                "ref_audio_path",
                "prompt_lang",
                "prompt_text",
                "media_type",
                "streaming_mode",
            }
            for k in allow_keys:
                if k in tts_params and tts_params[k] is not None:
                    payload[k] = tts_params[k]

        if not payload["ref_audio_path"]:
            return {"ok": False, "error": "TTS_REF_AUDIO_PATH is empty", "filename": None, "path": None}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}

        print(f"[TTS] POST {url}")
        print(f"[TTS] ref_audio_path={payload['ref_audio_path']}")
        print(f"[TTS] text_len={len(clean_text)}")

        try:
            try:
                # ✅ 用 json=payload 更稳，不容易出现编码/Content-Type 不一致导致的 400
                r = requests.post(url, json=payload, timeout=120)
                if r.status_code != 200:
                    # ✅ 把服务端返回内容带出来（这就是你现在缺的关键信息）
                    err_text = (r.text or "").strip()
                    return {
                        "ok": False,
                        "error": f"HTTP {r.status_code}: {err_text[:800]}",
                        "filename": None,
                        "path": None,
                    }
            except Exception as e:
                return {"ok": False, "error": f"request failed: {e}", "filename": None, "path": None}
        except Exception as e:
            return {"ok": False, "error": f"request failed: {e}", "filename": None, "path": None}

        os.makedirs("data/tts", exist_ok=True)
        filename = f"tts_{int(time.time() * 1000)}.wav"
        path = os.path.join("data", "tts", filename)

        with open(path, "wb") as f:
            f.write(r.content)

        return {"ok": True, "error": None, "filename": filename, "path": path}

    # ✅ 新增：TTS 流式 bytes（给 FastAPI StreamingResponse 用）
    def tts_stream_bytes(self, text: str, tts_params: Optional[Dict[str, Any]] = None) -> Iterator[bytes]:
        """
        流式：yield 音频 bytes chunk（后端以 StreamingResponse(audio/wav) 返回）
        兼容：streaming_mode=True + stream=1
        """
        clean_text = self._clean_text_for_tts(text)

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

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}

        with requests.post(url, data=body, headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=self.tts_stream_chunk_size):
                if chunk:
                    yield chunk
                    
    def _clean_tts_piece(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        # 去掉开头 @昵称
        t = re.sub(r"^@[^\s，,。:：!?！？]+\s*", "", t).strip()
        # 去掉纯标点/空白
        t = re.sub(r"^[\s，,。.!！？?；;]+|[\s，,。.!！？?；;]+$", "", t).strip()
        # 去掉常见emoji（简单版，避免某些实现 400）
        t = re.sub(r"[\U00010000-\U0010ffff]", "", t).strip()
        return t