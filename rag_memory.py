# rag_memory.py
# -*- coding: utf-8 -*-
"""
FAISS 版长期记忆（RAGMemoryManager）
===================================

把本次直播记忆/长期记忆写入本地向量索引，用语义检索召回。

与你现有 SmartVirtualAnchor 对齐的接口：
- RAGMemoryManager(persist_root, anchor_id, user_id, similarity_top_k=4)
- add_memory(text, metadata=None) -> memory_id (str)
- search(query, top_k=4) -> List[MemoryHit]  (每个 hit 有 memory_id/score/text/metadata)
- delete_memory(memory_id) -> bool

实现说明（工程取舍，尽量稳）：
- 记忆原文与 metadata 以 JSONL 形式持久化：<persist_dir>/memories.jsonl
- 向量索引使用 llama-index + FAISS（CPU）
- 删除采用“标记删除 + 重建索引”（对单机/单主播足够稳定；数据量很大时可优化）
- 如果依赖未安装，会抛出带提示的 ImportError

推荐安装（在你的 venv 里）：
  pip install -U llama-index
  pip install -U sentence-transformers
  pip install -U llama-index-embeddings-huggingface
  pip install -U llama-index-vector-stores-faiss faiss-cpu
"""

from __future__ import annotations

import os
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---- optional deps (raise friendly error on use) ----
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
except Exception as e:  # pragma: no cover
    _LLAMA_IMPORT_ERROR = e
    Document = None  # type: ignore
    VectorStoreIndex = None  # type: ignore
    Settings = None  # type: ignore
    StorageContext = None  # type: ignore
    HuggingFaceEmbedding = None  # type: ignore
    FaissVectorStore = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    _FAISS_IMPORT_ERROR = e
    faiss = None  # type: ignore


@dataclass
class MemoryHit:
    memory_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class RAGMemoryManager:
    """
    单主播/单进程本地长期记忆：JSONL + FAISS 语义索引。
    """
    def __init__(
        self,
        persist_root: str,
        anchor_id: str,
        user_id: str,
        similarity_top_k: int = 4,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.persist_root = persist_root
        self.anchor_id = anchor_id
        self.user_id = user_id
        self.similarity_top_k = int(similarity_top_k)

        self.persist_dir = os.path.join(self.persist_root, self._safe(anchor_id), self._safe(user_id))
        os.makedirs(self.persist_dir, exist_ok=True)

        self.mem_path = os.path.join(self.persist_dir, "memories.jsonl")

        # default embedding model (small & common)
        self.embedding_model = embedding_model or os.environ.get(
            "RAG_EMBED_MODEL",
            "BAAI/bge-small-zh-v1.5",
        )

        self._ensure_deps()
        self._setup_embedder()
        self._load_memories()
        self._build_or_rebuild_index()

    # -------------------- internal helpers --------------------
    def _ensure_deps(self) -> None:
        if Document is None or VectorStoreIndex is None or Settings is None:
            raise ImportError(
                "llama-index 相关依赖未安装或导入失败。建议安装：\n"
                "  pip install -U llama-index sentence-transformers\n"
                "  pip install -U llama-index-embeddings-huggingface\n"
                "  pip install -U llama-index-vector-stores-faiss faiss-cpu\n"
                f"原始导入错误：{_LLAMA_IMPORT_ERROR}"
            )
        if faiss is None:
            raise ImportError(
                "FAISS 未安装或导入失败。建议安装：\n"
                "  pip install -U faiss-cpu\n"
                f"原始导入错误：{_FAISS_IMPORT_ERROR}"
            )

    def _setup_embedder(self) -> None:
        embed = HuggingFaceEmbedding(model_name=self.embedding_model)
        Settings.embed_model = embed
        self._embed = embed  # for dimension probing

    def _load_memories(self) -> None:
        self._memories: List[Dict[str, Any]] = []
        if not os.path.exists(self.mem_path):
            return
        with open(self.mem_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._memories.append(obj)
                except Exception:
                    continue

    def _save_memories(self) -> None:
        tmp = self.mem_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for m in self._memories:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        os.replace(tmp, self.mem_path)

    def _active_memories(self) -> List[Dict[str, Any]]:
        return [m for m in self._memories if not m.get("deleted")]

    def _probe_dim(self) -> int:
        v = self._embed.get_text_embedding("维度探针")  # type: ignore[attr-defined]
        return int(len(v))

    def _build_or_rebuild_index(self) -> None:
        dim = self._probe_dim()
        index = faiss.IndexFlatL2(dim)  # type: ignore[attr-defined]
        self._vector_store = FaissVectorStore(faiss_index=index)
        self._storage = StorageContext.from_defaults(vector_store=self._vector_store)

        docs: List[Document] = []
        for m in self._active_memories():
            meta = dict(m.get("metadata") or {})
            meta["memory_id"] = m["memory_id"]
            meta["anchor_id"] = self.anchor_id
            meta["user_id"] = self.user_id
            meta["created_at"] = m.get("created_at")
            docs.append(Document(text=m["text"], metadata=meta))

        if docs:
            self._index = VectorStoreIndex.from_documents(docs, storage_context=self._storage)
        else:
            self._index = VectorStoreIndex([], storage_context=self._storage)

    @staticmethod
    def _safe(s: str) -> str:
        s = (s or "").strip() or "default"
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)

    # -------------------- public API --------------------
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        t = (text or "").strip()
        if not t:
            raise ValueError("memory text is empty")

        memory_id = str(uuid.uuid4())
        obj = {
            "memory_id": memory_id,
            "text": t,
            "metadata": metadata or {},
            "created_at": time.time(),
            "deleted": False,
        }
        self._memories.append(obj)

        with open(self.mem_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        self._build_or_rebuild_index()
        return memory_id

    def search(self, query: str, top_k: int = 4) -> List[MemoryHit]:
        q = (query or "").strip()
        if not q:
            return []

        k = int(top_k) if top_k else self.similarity_top_k
        try:
            retriever = self._index.as_retriever(similarity_top_k=k)
            nodes = retriever.retrieve(q)
        except Exception:
            return []

        out: List[MemoryHit] = []
        for n in nodes:
            try:
                node = n.node
                score = float(getattr(n, "score", 0.0) or 0.0)
                meta = dict(getattr(node, "metadata", {}) or {})
                mid = str(meta.get("memory_id") or "")
                text = str(getattr(node, "text", "") or "")
                out.append(MemoryHit(memory_id=mid, score=score, text=text, metadata=meta))
            except Exception:
                continue
        return out

    def delete_memory(self, memory_id: str) -> bool:
        mid = (memory_id or "").strip()
        if not mid:
            return False
        found = False
        for m in self._memories:
            if m.get("memory_id") == mid and not m.get("deleted"):
                m["deleted"] = True
                m["deleted_at"] = time.time()
                found = True
                break
        if not found:
            return False

        self._save_memories()
        self._build_or_rebuild_index()
        return True
