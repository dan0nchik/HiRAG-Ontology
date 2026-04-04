"""Hybrid entity retrieval: BM25 + Vector + PageRank with RRF fusion.

This module provides a parameter-free hybrid retrieval layer for HiRAG.
Instead of relying solely on vector similarity to find relevant entities,
it combines three complementary signals via Reciprocal Rank Fusion (RRF):

- **Vector similarity** (semantic match via embeddings)
- **BM25** (lexical/keyword match on entity names + descriptions)
- **PageRank** (graph-structural importance)

RRF requires no learned weights and is robust across domains.

Additionally provides **MMR (Maximal Marginal Relevance)** reranking to
balance relevance and diversity among selected entities.
"""

import logging
import os
import json
import pickle
from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 index over entity names + descriptions
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercased tokenizer."""
    return text.lower().split()


class BM25Index:
    """Wrapper around rank_bm25.BM25Okapi for entity retrieval.

    Each document corresponds to one entity in the knowledge graph,
    constructed as "{entity_name} {description}".
    """

    def __init__(self):
        self._entity_names: list[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def build(self, entities: dict[str, dict]):
        """Build the index from a {entity_name: node_data} mapping."""
        self._entity_names = []
        corpus = []

        for name, data in entities.items():
            description = data.get("description", "") if isinstance(data, dict) else ""
            doc_text = f"{name} {description}"
            self._entity_names.append(name)
            corpus.append(_tokenize(doc_text))

        self._bm25 = BM25Okapi(corpus)

    def query(self, query_text: str, top_k: int = 200) -> list[dict]:
        """Return top-k entities ranked by BM25 score."""
        if self._bm25 is None or not self._entity_names:
            return []

        q_tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append({
                "entity_name": self._entity_names[idx],
                "bm25_score": float(scores[idx]),
            })
        return results

    def save(self, path: str):
        data = {
            "entity_names": self._entity_names,
            "bm25": self._bm25,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._entity_names = data["entity_names"]
        idx._bm25 = data["bm25"]
        return idx


# ---------------------------------------------------------------------------
# PageRank cache
# ---------------------------------------------------------------------------

def compute_pagerank(graph: nx.Graph, alpha: float = 0.85) -> dict[str, float]:
    """Compute PageRank scores for all nodes in the graph."""
    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph, alpha=alpha)


def save_pagerank(scores: dict[str, float], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False)


def load_pagerank(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *rank_lists: list[dict],
    k: int = 60,
    entity_name_key: str = "entity_name",
    top_n: int = 200,
) -> list[dict]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion (RRF).

    The constant *k* (default 60) is from the original RRF paper
    (Cormack, Clarke & Buettcher, 2009).
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    entity_data: dict[str, dict] = {}

    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list):
            name = item[entity_name_key]
            rrf_scores[name] += 1.0 / (k + rank + 1)
            if name not in entity_data or len(str(item)) > len(str(entity_data[name])):
                entity_data[name] = item

    sorted_names = sorted(rrf_scores.keys(), key=lambda n: rrf_scores[n], reverse=True)

    results = []
    for name in sorted_names[:top_n]:
        entry = {**entity_data[name], "rrf_score": rrf_scores[name]}
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# High-level hybrid query function
# ---------------------------------------------------------------------------

async def hybrid_entity_retrieval(
    query: str,
    entities_vdb,
    bm25_index: Optional[BM25Index],
    pagerank_scores: Optional[dict[str, float]],
    top_k: int = 200,
    rrf_k: int = 60,
) -> list[dict]:
    """Retrieve entities using hybrid RRF fusion of vector, BM25, and PageRank.

    Falls back gracefully: if bm25_index or pagerank_scores are None,
    only the available signals are fused.
    """
    rank_lists = []

    # 1. Vector similarity (always available)
    vector_results = await entities_vdb.query(query, top_k=top_k)
    vector_ranked = [
        {"entity_name": r["entity_name"], "distance": r.get("distance", 0)}
        for r in vector_results
    ]
    rank_lists.append(vector_ranked)

    # 2. BM25 (if index is available)
    if bm25_index is not None:
        bm25_results = bm25_index.query(query, top_k=top_k)
        rank_lists.append(bm25_results)

    # 3. PageRank (static, query-independent ranking)
    if pagerank_scores:
        pr_sorted = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        pr_ranked = [
            {"entity_name": name, "pagerank": score}
            for name, score in pr_sorted[:top_k]
        ]
        rank_lists.append(pr_ranked)

    if len(rank_lists) == 1:
        return vector_results

    return reciprocal_rank_fusion(*rank_lists, k=rrf_k, top_n=top_k)


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) reranking
# ---------------------------------------------------------------------------

def mmr_rerank(
    candidates: list[dict],
    query_emb: np.ndarray,
    lam: float = 0.7,
    top_k: int = 20,
) -> list[dict]:
    """Select top_k entities from candidates balancing relevance and diversity.

    MMR score = λ · sim(candidate, query) − (1−λ) · max sim(candidate, selected)

    Each candidate must have a '__vector__' key with its embedding.
    Candidates without '__vector__' are appended at the end (fallback).

    Args:
        candidates: entity dicts, each with '__vector__' np.ndarray
        query_emb: query embedding vector
        lam: trade-off (1.0 = pure relevance, 0.0 = pure diversity)
        top_k: number of entities to select
    """
    if len(candidates) <= top_k:
        return list(candidates)

    # separate candidates with/without embeddings
    with_emb = [(i, c) for i, c in enumerate(candidates) if c.get("__vector__") is not None]
    without_emb = [c for c in candidates if c.get("__vector__") is None]

    if not with_emb:
        return candidates[:top_k]

    indices = [i for i, _ in with_emb]
    emb_matrix = np.stack([candidates[i]["__vector__"] for i in indices]).astype(np.float32)

    # normalise for cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normed = emb_matrix / norms

    q_norm = np.linalg.norm(query_emb)
    q_normed = (query_emb / q_norm).astype(np.float32) if q_norm > 0 else query_emb.astype(np.float32)

    # relevance scores: cosine(candidate, query)
    relevance = emb_normed @ q_normed  # shape (n,)

    selected: list[int] = []       # indices into emb_normed
    remaining = set(range(len(indices)))

    for _ in range(min(top_k, len(indices))):
        best_score = -float("inf")
        best_local = -1

        for local_idx in remaining:
            rel = float(relevance[local_idx])
            if selected:
                sims = emb_normed[selected] @ emb_normed[local_idx]
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0
            score = lam * rel - (1.0 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_local = local_idx

        if best_local < 0:
            break
        selected.append(best_local)
        remaining.discard(best_local)

    result = [candidates[indices[s]] for s in selected]

    # fill remaining slots from without_emb if needed
    if len(result) < top_k:
        result.extend(without_emb[: top_k - len(result)])

    return result
