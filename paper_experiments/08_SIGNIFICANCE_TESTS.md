# Statistical Significance Tests

All tests computed from LLM-as-judge (GPT-4o) evaluations on 130 queries
from UltraDomain Mix. Each query judged twice with answer order swapped
(260 judge records per experiment). Ties excluded from win counts.

## Method

- **Binomial test** (two-sided): H₀: P(improved wins) = 0.5
- **Wilson score interval**: 95% confidence interval for win rate
- **Cohen's h**: effect size for two proportions (|h| > 0.2 = small, > 0.5 = medium, > 0.8 = large)
- Significance threshold: α = 0.05 (marked with *), α = 0.01 (marked with **), α = 0.001 (marked with ***)

## Hybrid RRF (k=20) vs Vector-only

Comparison: **hybrid_rrf** vs **vector_only**
Judge records: 260 (from 130 queries × 2 orderings)

| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| Comprehensiveness | 139 | 121 | 0 | 0.535 | [0.474, 0.594] | 0.2917 | ns | +0.069 |
| Empowerment | 142 | 118 | 0 | 0.546 | [0.485, 0.606] | 0.1536 | ns | +0.092 |
| Diversity | 140 | 120 | 0 | 0.538 | [0.478, 0.598] | 0.2386 | ns | +0.077 |
| Overall Winner | 142 | 118 | 0 | 0.546 | [0.485, 0.606] | 0.1536 | ns | +0.092 |

## Vector+BM25 vs Vector-only

Comparison: **vector_bm25** vs **vector_only**
Judge records: 260 (from 130 queries × 2 orderings)

| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| Comprehensiveness | 136 | 124 | 0 | 0.523 | [0.462, 0.583] | 0.4952 | ns | +0.046 |
| Empowerment | 136 | 124 | 0 | 0.523 | [0.462, 0.583] | 0.4952 | ns | +0.046 |
| Diversity | 130 | 130 | 0 | 0.500 | [0.440, 0.560] | 1.0000 | ns | +0.000 |
| Overall Winner | 134 | 126 | 0 | 0.515 | [0.455, 0.575] | 0.6643 | ns | +0.031 |

## Vector+PageRank vs Vector-only

Comparison: **vector_pr** vs **vector_only**
Judge records: 260 (from 130 queries × 2 orderings)

| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| Comprehensiveness | 128 | 132 | 0 | 0.492 | [0.432, 0.553] | 0.8524 | ns | -0.015 |
| Empowerment | 131 | 129 | 0 | 0.504 | [0.443, 0.564] | 0.9506 | ns | +0.008 |
| Diversity | 124 | 136 | 0 | 0.477 | [0.417, 0.538] | 0.4952 | ns | -0.046 |
| Overall Winner | 132 | 128 | 0 | 0.508 | [0.447, 0.568] | 0.8524 | ns | +0.015 |

## Vector+BM25+PR vs Vector-only

Comparison: **vector_bm25_pr** vs **vector_only**
Judge records: 260 (from 130 queries × 2 orderings)

| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| Comprehensiveness | 149 | 111 | 0 | 0.573 | [0.512, 0.632] | 0.0216 | * | +0.147 |
| Empowerment | 140 | 120 | 0 | 0.538 | [0.478, 0.598] | 0.2386 | ns | +0.077 |
| Diversity | 146 | 114 | 0 | 0.562 | [0.501, 0.621] | 0.0543 | ns | +0.123 |
| Overall Winner | 143 | 117 | 0 | 0.550 | [0.489, 0.609] | 0.1209 | ns | +0.100 |

## Hybrid RRF (k=10) vs Vector-only (k=20)

Comparison: **hybrid_k10** vs **vector_k20**
Judge records: 260 (from 130 queries × 2 orderings)

| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| Comprehensiveness | 147 | 113 | 0 | 0.565 | [0.505, 0.624] | 0.0405 | * | +0.131 |
| Empowerment | 159 | 101 | 0 | 0.612 | [0.551, 0.669] | 0.0004 | *** | +0.225 |
| Diversity | 144 | 116 | 0 | 0.554 | [0.493, 0.613] | 0.0938 | ns | +0.108 |
| Overall Winner | 152 | 108 | 0 | 0.585 | [0.524, 0.643] | 0.0075 | ** | +0.170 |

## Hybrid RRF (k=7) vs Vector-only (k=20)

Comparison: **hybrid_k7** vs **vector_k20**
Judge records: 260 (from 130 queries × 2 orderings)

| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| Comprehensiveness | 145 | 115 | 0 | 0.558 | [0.497, 0.617] | 0.0719 | ns | +0.116 |
| Empowerment | 142 | 118 | 0 | 0.546 | [0.485, 0.606] | 0.1536 | ns | +0.092 |
| Diversity | 139 | 121 | 0 | 0.535 | [0.474, 0.594] | 0.2917 | ns | +0.069 |
| Overall Winner | 141 | 119 | 0 | 0.542 | [0.482, 0.602] | 0.1927 | ns | +0.085 |

## MMR λ=0.7 vs Hybrid baseline

Comparison: **mmr_07** vs **hybrid_baseline**
Judge records: 260 (from 130 queries × 2 orderings)

| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| Comprehensiveness | 119 | 141 | 0 | 0.458 | [0.398, 0.518] | 0.1927 | ns | -0.085 |
| Empowerment | 124 | 136 | 0 | 0.477 | [0.417, 0.538] | 0.4952 | ns | -0.046 |
| Diversity | 137 | 123 | 0 | 0.527 | [0.466, 0.587] | 0.4202 | ns | +0.054 |
| Overall Winner | 124 | 136 | 0 | 0.477 | [0.417, 0.538] | 0.4952 | ns | -0.046 |

## MMR λ=0.9 vs Hybrid baseline

Comparison: **mmr_09** vs **hybrid_baseline**
Judge records: 260 (from 130 queries × 2 orderings)

| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|---|
| Comprehensiveness | 123 | 137 | 0 | 0.473 | [0.413, 0.534] | 0.4202 | ns | -0.054 |
| Empowerment | 127 | 133 | 0 | 0.488 | [0.428, 0.549] | 0.7566 | ns | -0.023 |
| Diversity | 123 | 137 | 0 | 0.473 | [0.413, 0.534] | 0.4202 | ns | -0.054 |
| Overall Winner | 127 | 133 | 0 | 0.488 | [0.428, 0.549] | 0.7566 | ns | -0.023 |

## Summary: Overall Winner Significance

| Experiment | Win rate | 95% CI | p-value | Significant? | Effect size |
|---|---|---|---|---|---|
| Hybrid RRF (k=20) vs Vector-only | 0.546 | [0.485, 0.606] | 0.1536 | No | +0.092 (negligible) |
| Vector+BM25 vs Vector-only | 0.515 | [0.455, 0.575] | 0.6643 | No | +0.031 (negligible) |
| Vector+PageRank vs Vector-only | 0.508 | [0.447, 0.568] | 0.8524 | No | +0.015 (negligible) |
| Vector+BM25+PR vs Vector-only | 0.550 | [0.489, 0.609] | 0.1209 | No | +0.100 (negligible) |
| Hybrid RRF (k=10) vs Vector-only (k=20) | 0.585 | [0.524, 0.643] | 0.0075 | Yes (p<0.01) | +0.170 (negligible) |
| Hybrid RRF (k=7) vs Vector-only (k=20) | 0.542 | [0.482, 0.602] | 0.1927 | No | +0.085 (negligible) |
| MMR λ=0.7 vs Hybrid baseline | 0.477 | [0.417, 0.538] | 0.4952 | No | -0.046 (negligible) |
| MMR λ=0.9 vs Hybrid baseline | 0.488 | [0.428, 0.549] | 0.7566 | No | -0.023 (negligible) |

## Interpretation Guide

- **Win rate > 0.5**: improved variant wins more often
- **95% CI not containing 0.5**: result is significant at α=0.05
- **Cohen's h**: positive = improved wins more; |h| > 0.2 = practically meaningful
- **p-value**: probability of observing this result if both variants are equally good

## Script

```bash
cd eval/
python significance_tests.py                    # text output
python significance_tests.py --format markdown  # markdown tables
```
