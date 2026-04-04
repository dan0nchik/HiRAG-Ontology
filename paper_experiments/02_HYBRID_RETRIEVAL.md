# Experiment 1: Hybrid Entity Retrieval via RRF

## Motivation

HiRAG's entity retrieval (Equation 13 in original paper) relies solely on embedding cosine similarity — the same mechanism as NaiveRAG, just applied over entity names instead of text chunks. This is identified as a key limitation (Section 6 of original paper: "relies solely on LLM-generated weights for relation ranking").

We propose fusing three complementary signals via Reciprocal Rank Fusion (RRF):

1. **Vector similarity** (semantic) — existing HiRAG signal
2. **BM25** (lexical/keyword) — captures exact name matches missed by embeddings
3. **PageRank** (graph-structural) — favors well-connected, structurally important entities

RRF is parameter-free: `score(e) = Σ 1/(k + rank_i(e))` where k=60 (Cormack et al., 2009).

## Setup

- **Baseline**: Vector-only entity retrieval (original HiRAG)
- **Hybrid**: Vector + BM25 + PageRank fused via RRF
- **top_k**: 20 (default)
- **Queries**: 130 from UltraDomain Mix
- **Judge**: GPT-4o, 260 records (each query judged twice with swapped order)

## Results

### Win Rates (130 queries, 260 judge records)

| Criterion | Baseline (Vector) | Hybrid (RRF) | Δ |
|---|---|---|---|
| Comprehensiveness | 0.465 | **0.535** | +7.0% |
| Empowerment | 0.454 | **0.546** | +9.2% |
| Diversity | 0.462 | **0.538** | +7.7% |
| **Overall Winner** | **0.454** | **0.546** | **+9.2%** |

Hybrid wins all four criteria. Overall Winner: **54.6% vs 45.4%**.

### Context Token Statistics

| Metric | Baseline | Hybrid | Δ |
|---|---|---|---|
| Mean tokens | 10,419 | 21,197 | +10,778 (+103.4%) |
| Median tokens | 10,890 | 21,570 | +10,680 |
| Std dev | 5,677 | 3,746 | — |
| Contexts changed | — | 130/130 | 100% |
| Queries with more tokens (hybrid) | — | 130/130 | 100% |

The hybrid method consistently retrieves more diverse entities, leading to more communities and text units in context.

### Answer Token Statistics

| Metric | Baseline | Hybrid |
|---|---|---|
| Mean answer tokens | ~343 | ~404 |

Hybrid answers are slightly longer (+18%), suggesting the additional context enables more comprehensive responses.

## Implementation Details

- BM25 index: `rank_bm25.BM25Okapi` over "{entity_name} {description}" documents
- PageRank: `networkx.pagerank(alpha=0.85)` over the full entity-relation graph
- Both indexes computed once after graph construction, persisted to disk
- RRF constant k=60 (from original paper, no tuning)
- No additional LLM calls during retrieval — only one embedding call for query vector

## Files

- Script: `eval/hybrid_retrieval_eval.py`
- Summary: `eval/datasets/mix/mix_hi_hybrid_retrieval_q130_summary.json`
- Metrics: `eval/datasets/mix/mix_hi_hybrid_retrieval_q130_metrics.jsonl`
- Answers: `eval/datasets/mix/mix_hi_hybrid_retrieval_q130_answers_{baseline,hybrid}.jsonl`
- Judge: `eval/datasets/mix/mix_hi_hybrid_retrieval_q130_judge_result_openai.jsonl`
