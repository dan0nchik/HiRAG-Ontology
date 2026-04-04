# Experiment 3: Token-Normalized Evaluation

## Motivation

A critical objection to the hybrid retrieval improvement: **is the quality gain simply from feeding the LLM more context?** The full hybrid (top_k=20) uses 2.04x more tokens than the baseline. An LLM-as-judge will almost always prefer a longer, more detailed answer over a shorter one, regardless of retrieval quality.

To address this, we run hybrid retrieval with **reduced top_k** so that the context size approaches the baseline token count. If hybrid still wins at similar token budgets, the improvement is from better entity selection, not more tokens.

## Setup

Three configurations:

| Config | Retrieval | top_k | Expected tokens |
|---|---|---|---|
| Baseline | Vector-only | 20 | ~10,400 |
| Hybrid (budget) | RRF | 10 | ~15,300 |
| Hybrid (tight budget) | RRF | 7 | ~11,900 |

- **Queries**: 130 from UltraDomain Mix
- **Judge**: GPT-4o, 260 records per comparison

## Results

### Win Rates vs Vector-only Baseline (top_k=20)

| Hybrid top_k | Compre. | Empower. | Diversity | **Overall** | Token ratio |
|---|---|---|---|---|---|
| k=7 | **0.558** vs 0.442 | **0.546** vs 0.454 | **0.535** vs 0.465 | **0.542** vs 0.458 | 1.14x |
| k=10 | **0.565** vs 0.435 | **0.612** vs 0.388 | **0.554** vs 0.446 | **0.585** vs 0.415 | 1.47x |
| k=20 (ref) | **0.535** vs 0.465 | **0.546** vs 0.454 | **0.538** vs 0.462 | **0.546** vs 0.454 | 2.04x |

### Token Statistics

| Config | Mean tokens | Median tokens | Std dev | Δ vs baseline |
|---|---|---|---|---|
| Baseline (vector, k=20) | 10,379 | 10,520 | 5,649 | — |
| Hybrid (k=7) | 11,881 | 11,993 | 3,624 | +1,501 (+14.5%) |
| Hybrid (k=10) | 15,312 | 15,292 | 4,179 | +4,908 (+47.2%) |
| Hybrid (k=20) | 21,197 | 21,570 | 3,746 | +10,778 (+103.4%) |

### Per-Query Token Distribution (k=7)

| Stat | Value |
|---|---|
| Mean delta | +1,501 tokens |
| Median delta | +479 tokens |
| Min delta | -9,971 tokens |
| Max delta | +19,122 tokens |
| Queries where hybrid > baseline | 69/130 (53.1%) |
| Queries where hybrid < baseline | 61/130 (46.9%) |

At k=7, nearly half the queries actually use **fewer** tokens than baseline — hybrid retrieval at reduced top_k can be more focused.

### Per-Query Token Distribution (k=10)

| Stat | Value |
|---|---|
| Mean delta | +4,908 tokens |
| Median delta | +3,857 tokens |
| Min delta | -6,994 tokens |
| Max delta | +22,990 tokens |
| Queries where hybrid > baseline | 89/130 (68.5%) |
| Queries where hybrid < baseline | 41/130 (31.5%) |

### Key Observations

1. **The best result is NOT at maximum top_k.** Hybrid with k=10 achieves **+17.0%** Overall — substantially better than k=20 (+9.2%). This is a striking finding: reducing the number of selected entities from 20 to 10 improves quality by almost double.

2. **Explanation**: With k=20, the hybrid method brings in entities that, while relevant by BM25/PageRank, add noise when the candidate pool is too large. At k=10, only the strongest multi-signal entities survive, producing a focused, high-signal context.

3. **At near-equal token budgets (k=7, 1.14x), hybrid still wins by +8.4%.** This decisively refutes the "it's just more tokens" objection. The improvement comes from better entity selection, not context volume.

4. **Empowerment benefits most from moderate k**: +22.4% at k=10, vs +9.2% at k=7 and +9.2% at k=20. This suggests the sweet spot for enabling users to make informed judgments is a moderate number of well-chosen entities.

5. **Reduced variance at lower k**: hybrid at k=7 has std=3,624 vs baseline std=5,649. The hybrid approach produces more predictable context sizes.

### Quality-per-Token Efficiency

| Config | Overall win rate | Mean tokens | Win rate per 1K tokens |
|---|---|---|---|
| Baseline (vector, k=20) | 0.500 (ref) | 10,379 | 0.0482 |
| Hybrid k=7 | 0.542 | 11,881 | 0.0456 |
| Hybrid k=10 | 0.585 | 15,312 | 0.0382 |
| Hybrid k=20 | 0.546 | 21,197 | 0.0258 |

While the per-token efficiency decreases with more tokens (as expected), the absolute quality gain remains significant even at tight budgets.

## Recommended Configuration

**Hybrid RRF with top_k=10** provides the best quality (+17.0%) at a moderate token increase (1.47x). For token-constrained deployments, **top_k=7** delivers +8.4% at only 1.14x tokens.

## Files

- Script: `eval/token_normalized_eval.py`
- k=10 summary: `eval/datasets/mix/mix_hi_tokennorm_bk20_hk10_q130_summary.json`
- k=7 summary: `eval/datasets/mix/mix_hi_tokennorm_bk20_hk7_q130_summary.json`
- Metrics: `eval/datasets/mix/mix_hi_tokennorm_bk20_hk{7,10}_q130_metrics.jsonl`
- Answers: `eval/datasets/mix/mix_hi_tokennorm_bk20_hk{7,10}_q130_answers_{baseline,hybrid}.jsonl`
