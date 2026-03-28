"""Hybrid entity retrieval: BM25 + Vector + PageRank with RRF fusion.

This module provides a parameter-free hybrid retrieval layer for HiRAG.
Instead of relying solely on vector similarity to find relevant entities,
it combines three complementary signals via Reciprocal Rank Fusion (RRF):

- **Vector similarity** (semantic match via embeddings)
- **BM25** (lexical/keyword match on entity names + descriptions)
- **PageRank** (graph-structural importance)

RRF requires no learned weights and is robust across domains.
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
