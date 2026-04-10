# Experiment 4: MMR Reranking (Negative Result)

## Motivation

Maximal Marginal Relevance (MMR) is a classic technique (Carbonell & Goldstein, 1998) for balancing relevance and diversity in information retrieval. The idea: after initial retrieval, iteratively select entities that are both relevant to the query AND different from already-selected entities.

MMR score = λ · sim(entity, query) − (1−λ) · max sim(entity, selected)

We hypothesized that MMR could improve HiRAG's entity selection by reducing redundancy among the top-k selected entities, potentially improving the Diversity criterion.

## Setup

- **Baseline**: Hybrid retrieval (RRF) with standard top-k=20 selection
- **MMR**: Hybrid retrieval (RRF) with MMR reranking at the entity selection step
- **Lambda values tested**: 0.7 (moderate diversity) and 0.9 (mostly relevance)
- **Queries**: 130 from UltraDomain Mix
- **Judge**: GPT-4o, 260 records per comparison

Entity embeddings for MMR were retrieved directly from NanoVectorDB's internal storage — no additional embedding API calls required.

## Results

### Win Rates: Hybrid Baseline vs Hybrid+MMR

| λ | Compre. | Empower. | Diversity | **Overall** |
|---|---|---|---|---|
| 0.7 | 0.458 vs **0.542** | 0.477 vs **0.523** | **0.527** vs 0.473 | 0.477 vs **0.523** |
| 0.9 | 0.473 vs **0.527** | 0.488 vs **0.512** | 0.473 vs **0.527** | 0.488 vs **0.512** |

**Note**: Values shown as `MMR vs Baseline`. Baseline wins Overall in both cases.

Corrected (Baseline vs MMR perspective):

| λ | Compre. (baseline) | Empower. (baseline) | Diversity (baseline) | **Overall (baseline)** |
|---|---|---|---|---|
| 0.7 | **0.542** | **0.523** | 0.473 | **0.523** |
| 0.9 | **0.527** | **0.512** | **0.527** | **0.512** |

MMR **loses** on Overall Winner in both configurations.

### Token Statistics

| Config | Mean context tokens | Δ vs baseline |
|---|---|---|
| Baseline (hybrid, k=20) | 21,194 | — |
| MMR λ=0.7 | 19,414 | -1,780 (-8.4%) |
| MMR λ=0.9 | 16,421 | -4,772 (-22.5%) |

### Detailed Per-Criterion Analysis

**λ=0.7 (moderate diversity)**:
- Comprehensiveness: baseline wins 0.542 vs 0.458 (-8.4%)
- Empowerment: baseline wins 0.523 vs 0.477 (-4.6%)
- **Diversity: MMR wins 0.527 vs 0.473 (+5.4%)**
- Overall: baseline wins 0.523 vs 0.477 (-4.6%)

**λ=0.9 (mostly relevance)**:
- Comprehensiveness: baseline wins 0.527 vs 0.473 (-5.4%)
- Empowerment: baseline wins 0.512 vs 0.488 (-2.3%)
- Diversity: baseline wins 0.527 vs 0.473 (-5.4%)
- Overall: baseline wins 0.512 vs 0.488 (-2.3%)

### Context Token Details

| Metric | Baseline | MMR λ=0.7 | MMR λ=0.9 |
|---|---|---|---|
| Mean tokens | 21,194 | 19,414 | 16,421 |
| Median tokens | 21,535 | 20,744 | 16,878 |
| Mean delta | — | -1,780 | -4,772 |
| Median delta | — | -1,071 | -4,252 |
| Contexts changed | — | 130/130 (100%) | 130/130 (100%) |

## Analysis: Why MMR Hurts

1. **HiRAG entities are already topically coherent.** The Mix dataset has distinct documents on unrelated topics. For a query about topic X, the top-200 candidates are predominantly from document X. MMR tries to diversify among entities that are already from the same topic — it pushes out relevant entities and replaces them with tangentially related ones from other documents.

2. **Diversity at the entity level ≠ diversity at the answer level.** Selecting entities from diverse parts of the graph doesn't necessarily produce more diverse answers. In fact, it can fragment the context and reduce the LLM's ability to synthesize coherent reasoning.

3. **Token reduction is a side effect, not a feature.** MMR's more diverse entities span fewer overlapping communities and text units, reducing context size. But this reduction comes at the cost of coverage — the removed tokens were actually useful.

4. **Paradox of λ=0.9**: Higher λ (more relevance, less diversity) produces MORE token reduction (-22.5% vs -8.4%). This suggests that even small diversity pressure reorders entities significantly, and the re-ordered list maps to substantially different communities.

## Conclusion

MMR reranking is **not recommended** for HiRAG. The entity retrieval already produces a sufficiently focused set, and forcing diversity at the entity selection stage hurts comprehensiveness without meaningful gains.

However, the experiment provides valuable insight: **the bottleneck in HiRAG retrieval is entity identification quality, not entity diversity**. This motivates the hybrid retrieval approach (better signals for finding relevant entities) over post-hoc reranking (reorganizing already-found entities).

## Files

- Script: `eval/mmr_reranking_eval.py`
- λ=0.7 summary: `eval/datasets/mix/mix_hi_mmr_lam07_q130_summary.json`
- λ=0.9 summary: `eval/datasets/mix/mix_hi_mmr_lam09_q130_summary.json`
- Metrics: `eval/datasets/mix/mix_hi_mmr_lam{07,09}_q130_metrics.jsonl`
- Answers: `eval/datasets/mix/mix_hi_mmr_lam{07,09}_q130_answers_{baseline,mmr}.jsonl`
