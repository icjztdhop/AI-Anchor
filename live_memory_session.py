#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_memory_session.py
======================
单主播“上下文记忆 -> 本次直播记忆 -> 长期记忆”管线脚本（可独立跑，也可被服务端引入）。

规则：
1) 上下文记忆：对话消息列表（短期）
2) 当上下文超过模型 max_token（估算）时：
   - 将较早的上下文压缩成“本次直播记忆”（live_memory，按段落累积）
   - 上下文只保留最近 N 条消息继续对话
3) 直播结束时：把“本次直播记忆”写入长期记忆（RAGMemoryManager）

依赖：
- smart_anchor.py 里的 SmartVirtualAnchor
- （可选）rag_memory.py / llama-index：用于长期记忆；没有则只生成本次直播记忆文件

用法（交互式）：
  python live_memory_session.py --user_id room_001 --sender_name 路人甲
  输入弹幕后回车；输入 /end 结束直播并落长期记忆。
  输入 /stats 查看状态。

非交互式：
  python live_memory_session.py --lines "你好" "你们营业时间？" "再见" --end

可选：如果你想让“超限阈值”更小一点便于测试：
  python live_memory_session.py --max_ctx_tokens 400
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smart_anchor import SmartVirtualAnchor


def rough_token_estimate(text: str) -> int:
    """粗略估算 token（用于触发压缩阈值）"""
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_ctx_text(messages: List[Dict[str, Any]], anchor_name: str) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            sn = m.get("sender_name") or "观众"
            parts.append(f"{sn}：{content}")
        else:
            parts.append(f"{anchor_name}：{content}")
    return "\n".join(parts)


@dataclass
class LiveMemoryState:
    started_at: float = field(default_factory=time.time)
    chunks: List[str] = field(default_factory=list)
    total_chars: int = 0

    def add_chunk(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        self.chunks.append(t)
        self.total_chars += len(t)

    def to_text(self) -> str:
        head = f"【本次直播记忆】开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started_at))}\n"
        body = "\n\n".join(self.chunks).strip()
        return (head + body).strip()


class LiveMemoryManager:
    def __init__(
        self,
        anchor: SmartVirtualAnchor,
        user_id: str,
        sender_name: str = "观众",
        max_ctx_tokens: int = 1800,
        keep_last: int = 6,
        summary_temperature: float = 0.2,
    ) -> None:
        self.anchor = anchor
        self.user_id = user_id
        self.sender_name = sender_name
        self.max_ctx_tokens = max_ctx_tokens
        self.keep_last = keep_last
        self.summary_temperature = summary_temperature
        self.live = LiveMemoryState()

    def _summarize_to_live_memory(self, ctx_text: str) -> str:
        prompt = "\n\n".join([
            f"你是直播间虚拟主播{self.anchor.anchor_name}的后台总结助手。",
            "请把下面对话压缩成【本次直播记忆】的一段摘要，要求：",
            "1) 用中文，条理清晰，尽量简短但保留关键信息（人物、偏好、承诺、重要事实、待办）。",
            "2) 不要编造没出现的信息；不需要逐字复述。",
            "3) 建议用要点列表（每条一句话）。",
            "",
            "【需要压缩的对话】",
            ctx_text,
            "",
            "【输出：直播记忆摘要】",
        ]).strip()
        return self.anchor._call_lmdeploy(prompt, temperature=self.summary_temperature).strip()

    def maybe_compact_context(self) -> Optional[Dict[str, Any]]:
        session = self.anchor._get_session(self.user_id)
        msgs = session.memory.messages
        if not msgs:
            return None

        ctx_text = build_ctx_text(msgs, self.anchor.anchor_name)
        est = rough_token_estimate(ctx_text)

        if est <= self.max_ctx_tokens:
            return None
        if len(msgs) <= self.keep_last:
            return None

        to_compact = msgs[:-self.keep_last]
        remain = msgs[-self.keep_last:]

        compact_text = build_ctx_text(to_compact, self.anchor.anchor_name)
        summary = self._summarize_to_live_memory(compact_text)
        self.live.add_chunk(summary)

        # rolling_summary 叠加
        if session.memory.rolling_summary:
            session.memory.rolling_summary = (session.memory.rolling_summary.strip() + "\n" + summary).strip()
        else:
            session.memory.rolling_summary = summary

        session.memory.messages = remain

        return {
            "ok": True,
            "estimated_tokens_before": est,
            "compacted_messages": len(to_compact),
            "kept_messages": len(remain),
            "live_chunks": len(self.live.chunks),
            "live_total_chars": self.live.total_chars,
        }

    def chat_one(self, user_text: str) -> Dict[str, Any]:
        compact_before = self.maybe_compact_context()
        r = self.anchor.chat(user_text, user_id=self.user_id, sender_name=self.sender_name)
        compact_after = self.maybe_compact_context()
        return {
            "response": r["response"],
            "processing_time": r["processing_time"],
            "compact_before": compact_before,
            "compact_after": compact_after,
        }

    def end_live_and_persist(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        session = self.anchor._get_session(self.user_id)

        live_text = self.live.to_text()
        started = self.live.started_at
        ended = time.time()

        meta = {
            "type": "live_session",
            "anchor_name": self.anchor.anchor_name,
            "user_id": self.user_id,
            "started_at": started,
            "ended_at": ended,
            "duration_s": round(ended - started, 2),
        }
        if metadata:
            meta.update(metadata)

        os.makedirs("data/live_sessions", exist_ok=True)
        tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(started))
        live_path = os.path.join("data/live_sessions", f"live_memory_{self.user_id}_{tag}.txt")
        with open(live_path, "w", encoding="utf-8") as f:
            f.write(live_text + "\n")

        chat_path = os.path.join("data/live_sessions", f"chat_{self.user_id}_{tag}.json")
        with open(chat_path, "w", encoding="utf-8") as f:
            json.dump(session.memory.messages, f, ensure_ascii=False, indent=2)

        persisted = False
        memory_id = None
        error = None
        try:
            memory_id = self.anchor.add_long_term_memory(self.user_id, live_text, metadata=meta)
            persisted = True
        except Exception as e:
            error = str(e)

        return {
            "ok": True,
            "persisted": persisted,
            "memory_id": memory_id,
            "error": error,
            "live_file": live_path,
            "chat_file": chat_path,
            "live_chunks": len(self.live.chunks),
            "live_total_chars": self.live.total_chars,
            "meta": meta,
        }

    def stats(self) -> Dict[str, Any]:
        session = self.anchor._get_session(self.user_id)
        ctx_text = build_ctx_text(session.memory.messages, self.anchor.anchor_name)
        return {
            "ctx_messages": len(session.memory.messages),
            "ctx_est_tokens": rough_token_estimate(ctx_text),
            "max_ctx_tokens": self.max_ctx_tokens,
            "keep_last": self.keep_last,
            "rolling_summary_chars": len(session.memory.rolling_summary or ""),
            "live_chunks": len(self.live.chunks),
            "live_total_chars": self.live.total_chars,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", default=os.environ.get("USER_ID", "room_001"))
    ap.add_argument("--sender_name", default=os.environ.get("SENDER_NAME", "路人甲"))
    ap.add_argument("--lmdeploy_url", default=os.environ.get("LMDEPLOY_URL", "http://localhost:23333"))
    ap.add_argument("--knowledge_txt", default=os.environ.get("KNOWLEDGE_TXT", "knowledge.txt"))
    ap.add_argument("--max_ctx_tokens", type=int, default=int(os.environ.get("MAX_CTX_TOKENS", "1800")))
    ap.add_argument("--keep_last", type=int, default=int(os.environ.get("KEEP_LAST", "6")))
    ap.add_argument("--lines", nargs="*", default=None)
    ap.add_argument("--end", action="store_true")
    args = ap.parse_args()

    anchor = SmartVirtualAnchor(
        lmdeploy_url=args.lmdeploy_url,
        name=os.environ.get("ANCHOR_NAME", "小爱"),
        knowledge_txt_path=args.knowledge_txt,
    )

    mgr = LiveMemoryManager(
        anchor=anchor,
        user_id=args.user_id,
        sender_name=args.sender_name,
        max_ctx_tokens=args.max_ctx_tokens,
        keep_last=args.keep_last,
    )

    print("\n=== Live Memory Session ===")
    print(f"anchor={anchor.anchor_name} user_id={args.user_id} sender_name={args.sender_name}")
    print(f"max_ctx_tokens={args.max_ctx_tokens} keep_last={args.keep_last}")
    print("commands: /stats  /end\n")


    if args.lines:
        for i, line in enumerate(args.lines, 1):
            r = mgr.chat_one(line)
            print(f"\n[{i}] USER: {line}")
            if r.get("compact_before"):
                print(f"    [compact_before] {r['compact_before']}")
            print(f"    BOT: {r['response']}")
            if r.get("compact_after"):
                print(f"    [compact_after] {r['compact_after']}")
        if args.end:
            res = mgr.end_live_and_persist()
            print("\n=== END LIVE ===")
            print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    while True:
        try:
            line = input("YOU> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "/end"

        if not line:
            continue
        if line == "/stats":
            print(json.dumps(mgr.stats(), ensure_ascii=False, indent=2))
            continue
        if line == "/end":
            res = mgr.end_live_and_persist()
            print("\n=== END LIVE ===")
            print(json.dumps(res, ensure_ascii=False, indent=2))
            break

        r = mgr.chat_one(line)
        if r.get("compact_before"):
            print(f"[compact_before] {r['compact_before']}")
        print(f"BOT> {r['response']}")
        if r.get("compact_after"):
            print(f"[compact_after] {r['compact_after']}")


if __name__ == "__main__":
    main()
