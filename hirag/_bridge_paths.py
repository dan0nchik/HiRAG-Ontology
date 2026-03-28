"""Query-aware bridge path construction for HiRAG.

Instead of always taking the shortest path (minimum hops) between key entities,
this module provides multiple path-finding strategies that account for
**semantic relevance** of edges to the user query.

Strategies:
  1. shortest     – nx.shortest_path (baseline, hop-count only)
  2. dijkstra_qa  – Dijkstra with weight = 1 - cosine(edge, query)
  3. topk_rerank  – k-shortest simple paths, reranked by total relevance
  4. beam         – greedy beam search expanding top-B neighbours by relevance
  5. ppr          – Personalized PageRank seeded on key entities

Edge embeddings are pre-computed once at indexing time and stored on disk.
At query time only one embedding call (for the query) is needed.
"""

import logging
import os
import pickle
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _batch_cosine_similarity(query_emb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors."""
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    q_norm = np.linalg.norm(query_emb)
    if q_norm == 0:
        return np.zeros(len(matrix))
    return matrix @ query_emb / (norms * q_norm)


# ---------------------------------------------------------------------------
# Edge embeddings: compute, save, load
# ---------------------------------------------------------------------------

async def compute_edge_embeddings(
    graph: nx.Graph,
    embedding_func,
    batch_size: int = 64,
) -> dict[tuple[str, str], np.ndarray]:
    """Compute embeddings for all edge descriptions in the graph.

    Returns a dict mapping (src, tgt) sorted tuples to embedding vectors.
    Edges without a description are skipped.
    """
    edges = []
    texts = []
    for u, v, data in graph.edges(data=True):
        desc = data.get("description", "")
        if not desc:
            continue
        key = tuple(sorted((u, v)))
        edges.append(key)
        texts.append(desc)

    if not texts:
        return {}

    logger.info(f"Computing embeddings for {len(texts)} edges (batch_size={batch_size})")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = await embedding_func(batch)
        all_embeddings.extend(batch_embs)

    result = {}
    for key, emb in zip(edges, all_embeddings):
        result[key] = np.asarray(emb, dtype=np.float32)
    return result


def save_edge_embeddings(embs: dict[tuple[str, str], np.ndarray], path: str):
    """Save edge embeddings to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(embs, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved {len(embs)} edge embeddings to {path}")


def load_edge_embeddings(path: str) -> dict[tuple[str, str], np.ndarray]:
    """Load edge embeddings from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Edge relevance scoring
# ---------------------------------------------------------------------------

def _edge_relevance(
    u: str, v: str,
    query_emb: np.ndarray,
    edge_embeddings: dict[tuple[str, str], np.ndarray],
) -> float:
    """Return cosine similarity between query and edge embedding."""
    key = tuple(sorted((u, v)))
    emb = edge_embeddings.get(key)
    if emb is None:
        return 0.0
    return _cosine_similarity(query_emb, emb)


def _assign_query_weights(
    graph: nx.Graph,
    query_emb: np.ndarray,
    edge_embeddings: dict[tuple[str, str], np.ndarray],
):
    """Assign 'query_weight' = 1 - relevance to every edge in the graph.

    Lower weight → more relevant → preferred by Dijkstra.
    """
    for u, v, data in graph.edges(data=True):
        rel = _edge_relevance(u, v, query_emb, edge_embeddings)
        data["query_weight"] = 1.0 - rel


# ---------------------------------------------------------------------------
# Strategy 1: Shortest path (baseline)
# ---------------------------------------------------------------------------

def _find_path_shortest(graph: nx.Graph, key_entities: list[str]) -> list[str]:
    """Original HiRAG behaviour: shortest path through required nodes."""
    if len(key_entities) < 2:
        return list(key_entities)

    final_path = []
    current = key_entities[0]

    for next_node in key_entities[1:]:
        try:
            sub_path = nx.shortest_path(graph, source=current, target=next_node)
        except nx.NetworkXNoPath:
            final_path.append(next_node)
            current = next_node
            continue
        if final_path:
            final_path.extend(sub_path[1:])
        else:
            final_path.extend(sub_path)
        current = next_node

    return final_path


# ---------------------------------------------------------------------------
# Strategy 2: Dijkstra with query-aware edge weights
# ---------------------------------------------------------------------------

def _find_path_dijkstra_qa(
    graph: nx.Graph,
    key_entities: list[str],
    query_emb: np.ndarray,
    edge_embeddings: dict[tuple[str, str], np.ndarray],
) -> list[str]:
    """Dijkstra shortest path using weight = 1 - relevance(edge, query)."""
    _assign_query_weights(graph, query_emb, edge_embeddings)

    if len(key_entities) < 2:
        return list(key_entities)

    final_path = []
    current = key_entities[0]

    for next_node in key_entities[1:]:
        try:
            sub_path = nx.dijkstra_path(
                graph, source=current, target=next_node, weight="query_weight"
            )
        except nx.NetworkXNoPath:
            final_path.append(next_node)
            current = next_node
            continue
        if final_path:
            final_path.extend(sub_path[1:])
        else:
            final_path.extend(sub_path)
        current = next_node

    return final_path


# ---------------------------------------------------------------------------
# Strategy 3: Top-k shortest simple paths, reranked by total relevance
# ---------------------------------------------------------------------------

def _find_path_topk_rerank(
    graph: nx.Graph,
    key_entities: list[str],
    query_emb: np.ndarray,
    edge_embeddings: dict[tuple[str, str], np.ndarray],
    k: int = 5,
) -> list[str]:
    """Find k-shortest simple paths per segment, pick the one with highest
    total edge relevance to the query."""
    if len(key_entities) < 2:
        return list(key_entities)

    final_path = []
    current = key_entities[0]

    for next_node in key_entities[1:]:
        best_path = None
        best_score = -1.0

        try:
            candidates = list(
                nx.shortest_simple_paths(graph, current, next_node)
            )
        except nx.NetworkXNoPath:
            final_path.append(next_node)
            current = next_node
            continue

        for i, cand in enumerate(candidates):
            if i >= k:
                break
            score = 0.0
            for j in range(len(cand) - 1):
                score += _edge_relevance(cand[j], cand[j + 1], query_emb, edge_embeddings)
            if score > best_score:
                best_score = score
                best_path = cand

        if best_path is None:
            final_path.append(next_node)
            current = next_node
            continue

        if final_path:
            final_path.extend(best_path[1:])
        else:
            final_path.extend(best_path)
        current = next_node

    return final_path


# ---------------------------------------------------------------------------
# Strategy 4: Beam search (greedy, expands top-B neighbours by relevance)
# ---------------------------------------------------------------------------

def _find_path_beam(
    graph: nx.Graph,
    key_entities: list[str],
    query_emb: np.ndarray,
    edge_embeddings: dict[tuple[str, str], np.ndarray],
    beam_width: int = 3,
    max_depth: int = 10,
) -> list[str]:
    """Greedy beam search: at each step keep top-B partial paths ranked by
    cumulative edge relevance to the query."""
    if len(key_entities) < 2:
        return list(key_entities)

    final_path = []
    current = key_entities[0]

    for next_node in key_entities[1:]:
        segment = _beam_search_segment(
            graph, current, next_node, query_emb, edge_embeddings,
            beam_width, max_depth,
        )
        if segment is None:
            final_path.append(next_node)
            current = next_node
            continue
        if final_path:
            final_path.extend(segment[1:])
        else:
            final_path.extend(segment)
        current = next_node

    return final_path


def _beam_search_segment(
    graph: nx.Graph,
    source: str,
    target: str,
    query_emb: np.ndarray,
    edge_embeddings: dict[tuple[str, str], np.ndarray],
    beam_width: int,
    max_depth: int,
) -> Optional[list[str]]:
    """Beam search from source to target. Returns path or None."""
    if source == target:
        return [source]
    if source not in graph or target not in graph:
        return None

    # beams: list of (cumulative_score, path)
    beams = [(0.0, [source])]

    for _ in range(max_depth):
        candidates = []
        for score, path in beams:
            last = path[-1]
            if last == target:
                candidates.append((score, path))
                continue
            for neighbour in graph.neighbors(last):
                if neighbour in path:  # avoid cycles
                    continue
                edge_rel = _edge_relevance(last, neighbour, query_emb, edge_embeddings)
                candidates.append((score + edge_rel, path + [neighbour]))

        if not candidates:
            return None

        # separate completed and ongoing
        completed = [(s, p) for s, p in candidates if p[-1] == target]
        ongoing = [(s, p) for s, p in candidates if p[-1] != target]

        if completed:
            # return best completed path
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]

        # keep top beam_width
        ongoing.sort(key=lambda x: x[0], reverse=True)
        beams = ongoing[:beam_width]

    # max depth reached — check if any beam reached target
    for _, path in beams:
        if path[-1] == target:
            return path
    return None


# ---------------------------------------------------------------------------
# Strategy 5: Personalized PageRank
# ---------------------------------------------------------------------------

def _find_path_ppr(
    graph: nx.Graph,
    key_entities: list[str],
    alpha: float = 0.15,
    top_k: int = 30,
) -> list[str]:
    """Use Personalized PageRank seeded on key entities to find relevant
    intermediate nodes, then connect them via shortest paths."""
    if len(key_entities) < 2:
        return list(key_entities)

    # seed: uniform over key entities that exist in graph
    personalization = {}
    present = [e for e in key_entities if e in graph]
    if not present:
        return list(key_entities)
    for e in present:
        personalization[e] = 1.0 / len(present)

    try:
        ppr_scores = nx.pagerank(graph, alpha=alpha, personalization=personalization)
    except Exception:
        return _find_path_shortest(graph, key_entities)

    # top-k nodes by PPR score (always include key entities)
    sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
    top_nodes = set(present)
    for node, _ in sorted_nodes:
        if len(top_nodes) >= top_k:
            break
        top_nodes.add(node)

    subgraph = graph.subgraph(top_nodes)
    return _find_path_shortest(subgraph, key_entities)


# ---------------------------------------------------------------------------
# Edge pruning
# ---------------------------------------------------------------------------

def prune_edges_by_relevance(
    edges_data: list[dict],
    query_emb: np.ndarray,
    edge_embeddings: dict[tuple[str, str], np.ndarray],
    threshold: float = 0.1,
) -> list[dict]:
    """Remove edges from reasoning path whose relevance to query is below threshold.

    edges_data: list of dicts with 'src_tgt' key = (src, tgt) tuple.
    """
    if not edge_embeddings or threshold <= 0:
        return edges_data

    pruned = []
    for edge in edges_data:
        src_tgt = edge.get("src_tgt")
        if src_tgt is None:
            pruned.append(edge)
            continue
        rel = _edge_relevance(src_tgt[0], src_tgt[1], query_emb, edge_embeddings)
        if rel >= threshold:
            pruned.append(edge)

    removed = len(edges_data) - len(pruned)
    if removed > 0:
        logger.debug(f"Pruned {removed}/{len(edges_data)} edges below relevance {threshold}")
    return pruned


# ---------------------------------------------------------------------------
# Ordered unique key entities (replaces list(set(...)))
# ---------------------------------------------------------------------------

def ordered_unique(nested: list[list[str]]) -> list[str]:
    """Flatten nested entity name lists preserving ranking order."""
    seen = set()
    result = []
    for name in (k for kk in nested for k in kk):
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

STRATEGIES = {
    "shortest",
    "dijkstra_qa",
    "topk_rerank",
    "beam",
    "ppr",
}


def find_bridge_path(
    graph: nx.Graph,
    key_entities: list[str],
    query_emb: Optional[np.ndarray] = None,
    edge_embeddings: Optional[dict[tuple[str, str], np.ndarray]] = None,
    strategy: str = "dijkstra_qa",
    beam_width: int = 3,
    topk_paths: int = 5,
    ppr_alpha: float = 0.15,
) -> list[str]:
    """Find a bridge path through key_entities using the given strategy.

    Falls back to 'shortest' if query-aware data is missing.
    """
    if strategy not in STRATEGIES:
        logger.warning(f"Unknown bridge strategy '{strategy}', falling back to shortest")
        strategy = "shortest"

    # fallback if no embeddings for query-aware strategies
    needs_embs = strategy in ("dijkstra_qa", "topk_rerank", "beam")
    if needs_embs and (query_emb is None or not edge_embeddings):
        logger.warning(f"Missing embeddings for strategy '{strategy}', falling back to shortest")
        strategy = "shortest"

    if strategy == "shortest":
        return _find_path_shortest(graph, key_entities)
    elif strategy == "dijkstra_qa":
        return _find_path_dijkstra_qa(graph, key_entities, query_emb, edge_embeddings)
    elif strategy == "topk_rerank":
        return _find_path_topk_rerank(graph, key_entities, query_emb, edge_embeddings, k=topk_paths)
    elif strategy == "beam":
        return _find_path_beam(graph, key_entities, query_emb, edge_embeddings, beam_width=beam_width)
    elif strategy == "ppr":
        return _find_path_ppr(graph, key_entities, alpha=ppr_alpha)
    else:
        return _find_path_shortest(graph, key_entities)
