# Experiment 2: Ablation Study — RRF Signal Contributions

## Motivation

The full hybrid retrieval combines three signals (Vector + BM25 + PageRank). A natural question is: which signal contributes most? Is the improvement driven by BM25, PageRank, or the fusion itself?

## Setup

Four configurations, all compared against the **vector-only baseline**:

| Config | Vector | BM25 | PageRank |
|---|---|---|---|
| `vector_only` (baseline) | yes | — | — |
| `vector_bm25` | yes | yes | — |
| `vector_pr` | yes | — | yes |
| `vector_bm25_pr` (full) | yes | yes | yes |

- **Queries**: 130 from UltraDomain Mix
- **Judge**: GPT-4o, 260 records per comparison (3 comparisons = 780 total judge records)
- **top_k**: 20

## Results

### Pairwise Win Rates vs Vector-only Baseline

| Config | Compre. | Empower. | Diversity | **Overall** |
|---|---|---|---|---|
| Vector + BM25 | **0.523** vs 0.477 | **0.523** vs 0.477 | 0.500 vs 0.500 | **0.515** vs 0.485 |
| Vector + PageRank | 0.492 vs 0.508 | **0.504** vs 0.496 | 0.477 vs 0.523 | **0.508** vs 0.492 |
| **Vector + BM25 + PR** | **0.573** vs 0.427 | **0.538** vs 0.462 | **0.562** vs 0.438 | **0.550** vs 0.450 |

### Signal Contribution Summary

| Signal added | Overall Δ vs baseline | Dominant criterion |
|---|---|---|
| +BM25 alone | +3.1% | Comprehensiveness, Empowerment (tied at +4.6%) |
| +PageRank alone | +1.5% | Empowerment (+0.8%) |
| +BM25 + PageRank | **+10.0%** | Comprehensiveness (+14.6%) |

### Key Observations

1. **BM25 is the primary driver** (+3.1% alone). This makes sense: lexical matching catches entities with exact keyword matches that embedding similarity may miss (e.g., abbreviations, proper nouns with unusual spelling).

2. **PageRank alone is near-neutral** (+1.5%, within noise). As a query-independent signal, it biases toward well-connected hub nodes regardless of the query topic.

3. **Synergistic effect**: BM25 + PageRank together (+10.0%) substantially exceeds the sum of individual contributions (3.1% + 1.5% = 4.6%). The fusion via RRF creates value beyond what either signal provides alone — BM25 brings in lexically relevant entities that PageRank then properly ranks by structural importance.

4. **Diversity is driven by multi-signal fusion**: BM25 alone achieves 0.500 (neutral) on Diversity, PageRank alone scores 0.477 (slightly hurts), but together they score **0.562** (+12.4% over baseline). The RRF mechanism naturally diversifies the entity selection by combining differently-ordered ranked lists.

5. **Comprehensiveness benefits most** from the full hybrid: +14.6% gain. This suggests the additional entities retrieved via BM25/PageRank cover aspects that vector similarity alone misses.

### Mean Answer Tokens by Configuration

| Config | Mean answer tokens |
|---|---|
| vector_only | 343.1 |
| vector_bm25 | 370.7 |
| vector_pr | 359.4 |
| vector_bm25_pr | 403.9 |

The full hybrid produces ~18% longer answers, reflecting more comprehensive generation from richer context.

## Files

- Script: `eval/ablation_rrf_eval.py`
- Summary: `eval/datasets/mix/mix_hi_ablation_q130_summary.json`
- Answers: `eval/datasets/mix/mix_hi_ablation_q130_answers_{vector_only,vector_bm25,vector_pr,vector_bm25_pr}.jsonl`
- Judge files: `eval/datasets/mix/mix_hi_ablation_q130_judge_vector_only_vs_*.jsonl`
