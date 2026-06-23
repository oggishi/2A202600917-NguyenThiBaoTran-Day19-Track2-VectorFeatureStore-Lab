"""HybridMemoryAgent — episodic memory (Vector Store) + stable profile (Feast).

A minimal POC for the bonus challenge. Combines:
  - Episodic memory: per-user hybrid retrieval (BM25 + dense + RRF k=60) over a
    Qdrant in-memory collection, filtered by `user_id` payload. (lab §1–§3)
  - Stable profile + recent activity: Feast online lookup if the store has been
    materialized (app/feast_repo), else a deterministic synthetic fallback so the
    demo always runs. (lab §4 / §6)

recall() assembles a context string the way you'd feed an LLM prompt — no real
LLM call needed, per the brief. Optimised for clarity, not speed.

Run the demo:  python bonus/demo.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAST_DIR = REPO_ROOT / "app" / "feast_repo"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
COLLECTION = "episodic_memory"
RRF_K = 60
CHUNK_TARGET_TOKENS = 60   # sentence-aware target; see ARCHITECTURE.md decision 1

PROFILE_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]


@dataclass
class Memory:
    point_id: int
    user_id: str
    text: str


class HybridMemoryAgent:
    """Per-user hybrid memory: Qdrant (dense) + BM25 (sparse) + Feast (profile)."""

    def __init__(self, use_feast: bool = True) -> None:
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        self.memories: list[Memory] = []
        self._next_id = 0
        self._feast = self._connect_feast() if use_feast else None

    # ── Feast wiring (optional, graceful fallback) ──────────────────────
    def _connect_feast(self):
        try:
            from feast import FeatureStore

            fs = FeatureStore(repo_path=str(FEAST_DIR))
            # Probe once: if the online store isn't materialized this raises and
            # we fall back to synthetic features so the demo still exits 0.
            fs.get_online_features(
                features=PROFILE_FEATURES,
                entity_rows=[{"user_id": "u_001"}],
            ).to_dict()
            return fs
        except Exception as exc:  # noqa: BLE001
            print(f"[agent] Feast unavailable ({type(exc).__name__}); using synthetic profile.",
                  file=sys.stderr)
            return None

    def _get_profile(self, user_id: str) -> dict:
        if self._feast is not None:
            try:
                d = self._feast.get_online_features(
                    features=PROFILE_FEATURES,
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()
                return {k: v[0] for k, v in d.items()}
            except Exception:  # noqa: BLE001
                pass
        # Deterministic synthetic fallback (keeps demo runnable without materialize).
        i = int(re.sub(r"\D", "", user_id) or 0)
        topics = ["ai_ml", "cloud", "security", "database", "devops"]
        return {
            "user_id": user_id,
            "reading_speed_wpm": 180 + (i * 7) % 200,
            "preferred_language": "vi" if i % 3 != 0 else "en",
            "topic_affinity": topics[i % 5],
            "queries_last_hour": (i * 11) % 50,
            "distinct_topics_24h": 1 + (i * 3) % 10,
        }

    # ── chunking (decision 1: sentence-aware, VN-safe) ──────────────────
    @staticmethod
    def _chunk(text: str) -> list[str]:
        # Split on sentence punctuation / newlines — script-agnostic, never breaks
        # mid-word (Vietnamese words are multi-syllable; whitespace splitting would).
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        chunks: list[str] = []
        buf: list[str] = []
        n_tokens = 0
        for sent in sentences:
            buf.append(sent)
            n_tokens += len(sent.split())
            if n_tokens >= CHUNK_TARGET_TOKENS:
                chunks.append(" ".join(buf))
                buf, n_tokens = [], 0
        if buf:
            chunks.append(" ".join(buf))
        return chunks or [text.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user (synchronous)."""
        chunks = self._chunk(text)
        vectors = list(self.embedder.embed(chunks))
        points = []
        for chunk, vec in zip(chunks, vectors):
            mem = Memory(point_id=self._next_id, user_id=user_id, text=chunk)
            self.memories.append(mem)
            points.append(PointStruct(
                id=self._next_id,
                vector=vec.tolist(),
                payload={"user_id": user_id, "text": chunk},
            ))
            self._next_id += 1
        self.client.upsert(collection_name=COLLECTION, points=points)

    # ── retrieval ─────────────────────────────────────────────────────────
    def _search_semantic(self, query: str, user_id: str, top_k: int) -> list[str]:
        q_vec = next(self.embedder.embed([query])).tolist()
        res = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            limit=top_k,
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        )
        return [p.payload["text"] for p in res.points]

    def _search_keyword(self, query: str, user_id: str, top_k: int) -> list[str]:
        # BM25 over just this user's memories (privacy: never leak other users').
        own = [m.text for m in self.memories if m.user_id == user_id]
        if not own:
            return []
        bm25 = BM25Okapi([self._tokenize(t) for t in own])
        scores = bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(own)), key=lambda i: -scores[i])[:top_k]
        return [own[i] for i in ranked]

    def _search_hybrid(self, query: str, user_id: str, top_k: int = 3, rrf_k: int = RRF_K) -> list[str]:
        depth = max(top_k * 5, 20)
        kw = self._search_keyword(query, user_id, depth)
        sem = self._search_semantic(query, user_id, depth)
        rrf: dict[str, float] = {}
        for ids in (kw, sem):
            for rank, text in enumerate(ids, start=1):   # rank is 1-based
                rrf[text] = rrf.get(text, 0.0) + 1.0 / (rrf_k + rank)
        return [t for t, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]]

    # ── recall: assemble context (profile + recent + episodic) ──────────
    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve profile features + top-3 episodic memories → context string."""
        p = self._get_profile(user_id)
        top = self._search_hybrid(query, user_id, top_k=3)
        mem_block = "\n".join(f"    - {t}" for t in top) if top else "    (no memories yet)"
        return (
            f"[Context for {user_id}]\n"
            f"  Profile: likes '{p['topic_affinity']}', reads {p['reading_speed_wpm']} wpm, "
            f"language={p['preferred_language']}.\n"
            f"  Recent activity: {p['queries_last_hour']} queries/last-hour, "
            f"{p['distinct_topics_24h']} distinct topics/24h.\n"
            f"  Query: {query!r}\n"
            f"  Top memories (hybrid BM25+vector, RRF k={RRF_K}):\n{mem_block}"
        )


if __name__ == "__main__":
    # Tiny self-test so `python bonus/agent.py` is meaningful on its own.
    agent = HybridMemoryAgent()
    agent.remember("Tôi đã đọc về Kubernetes auto-scaling và quản lý vòng đời container.", "u_001")
    print(agent.recall("Tôi đã đọc gì về Kubernetes?", "u_001"))
