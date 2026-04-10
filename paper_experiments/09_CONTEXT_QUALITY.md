# Experiment 5: Context Quality Improvements

## Motivation

The hybrid RRF retrieval (Experiment 2-3) improved **entity selection** — finding better entities via multi-signal fusion. But the downstream pipeline still had inefficiencies:

1. **Generic generation prompt**: The LLM receives three-level hierarchical context (global communities, bridge paths, local entities, source evidence) but the prompt treats it as flat "data tables" with no guidance on how to use each level.
2. **Unranked source chunks**: Text chunks are selected by association with retrieved entities, not by relevance to the query. A chunk mentioning "AMAZON" in a logistics context is included even when the question is about cloud computing.
3. **Unfiltered community reports**: All community reports touching any retrieved entity are included, even tangentially related ones.

Key insight from prior experiments: **k=10 > k=20** — focused, high-quality context beats broad, noisy context. The same principle should apply to all context levels, not just entity selection.

## Improvements

### #6: Structured Generation Prompt

Replaced the generic `local_rag_response` prompt with `hierarchical_rag_response` that explicitly instructs the LLM on each context level:

- **GLOBAL CONTEXT (Backgrounds)**: "Use these to frame the overall scope of your answer"
- **BRIDGE CONTEXT (Reasoning Path)**: "Use these to explain HOW different topics relate"
- **LOCAL CONTEXT (Entity Details)**: "Use these for precise claims and specific details"
- **EVIDENCE (Source Documents)**: "Prefer claims that are directly supported by evidence here"

Cost: **0 additional tokens** (replaces existing prompt, roughly same length).

### #2: Source Chunk Reranking by Query Relevance

After collecting candidate text chunks via entity association, rerank them by cosine similarity between the query embedding and each chunk's embedding (first 500 chars). Most relevant evidence moves to the top of the token budget.

Cost: **1 batch embedding call** per query (~7 chunks × 500 chars).

### #3: Community Report Relevance Filtering

After collecting community reports via entity overlap, score each report by cosine similarity to the query (first 300 chars) and keep only the top-N most relevant. Removes tangential community reports that add noise.

Cost: **1 batch embedding call** per query (~5 reports × 300 chars).

## Setup

- **Baseline**: Hybrid RRF retrieval (k=10) with original prompt, no filtering
- **Improved**: Hybrid RRF retrieval (k=10) + all three improvements
- **Queries**: 130 from UltraDomain Mix
- **Judge**: GPT-4o, 260 records (each query judged twice with swapped order)
- **Graph**: Same cached graph, no reindexing

QueryParam configuration:

| Parameter | Baseline | Improved |
|---|---|---|
| `enable_hybrid_retrieval` | True | True |
| `top_k` | 10 | 10 |
| `use_structured_prompt` | False | True |
| `enable_chunk_reranking` | False | True |
| `enable_community_filtering` | False | True |
| `max_communities` | (unlimited) | 5 |

## Results

### Win Rates

| Criterion | Baseline | Improved | Delta |
|---|---|---|---|
| Comprehensiveness | 0.404 | **0.596** | **+19.2%** |
| Empowerment | 0.423 | **0.577** | **+15.4%** |
| Diversity | 0.358 | **0.642** | **+28.4%** |
| **Overall Winner** | 0.400 | **0.600** | **+20.0%** |

### Token Statistics

| Metric | Baseline | Improved | Delta |
|---|---|---|---|
| Mean context tokens | 15,304 | 15,262 | **-42** (-0.3%) |
| Contexts changed | — | 130/130 | 100% |

The improvement comes with **zero token increase** — in fact slightly fewer tokens due to community filtering.

### Statistical Significance

| Criterion | Improved wins | Baseline wins | Win rate | 95% CI | p-value | Sig. | Cohen's h |
|---|---|---|---|---|---|---|---|
| Comprehensiveness | 155 | 105 | 0.596 | [0.536, 0.654] | 0.0023 | ** | +0.194 |
| Empowerment | 150 | 110 | 0.577 | [0.516, 0.635] | 0.0154 | * | +0.154 |
| Diversity | 167 | 93 | 0.642 | [0.582, 0.698] | <0.0001 | *** | +0.289 |
| **Overall Winner** | **156** | **104** | **0.600** | **[0.539, 0.658]** | **0.0015** | **\*\*** | **+0.201** |

- **Overall Winner: p=0.0015** — statistically significant at p<0.01
- **Diversity: p<0.0001** — highly significant at p<0.001
- **Cohen's h = +0.201** — the only experiment across all our tests that reaches practically meaningful effect size (small, h>0.2)

### Comparison Across All Experiments

| Experiment | Overall win rate | p-value | Significant? | Cohen's h |
|---|---|---|---|---|
| Hybrid RRF k=20 vs Vector-only | 0.546 | 0.1536 | No | +0.092 |
| Vector+BM25 vs Vector-only | 0.515 | 0.6643 | No | +0.031 |
| Vector+PageRank vs Vector-only | 0.508 | 0.8524 | No | +0.015 |
| Vector+BM25+PR vs Vector-only | 0.550 | 0.1209 | No | +0.100 |
| **Hybrid RRF k=10 vs Vector-only k=20** | **0.585** | **0.0075** | **Yes (p<0.01)** | +0.170 |
| Hybrid RRF k=7 vs Vector-only k=20 | 0.542 | 0.1927 | No | +0.085 |
| MMR λ=0.7 vs Hybrid baseline | 0.477 | 0.4952 | No | -0.046 |
| MMR λ=0.9 vs Hybrid baseline | 0.488 | 0.7566 | No | -0.023 |
| **Context Quality vs Hybrid k=10** | **0.600** | **0.0015** | **Yes (p<0.01)** | **+0.201** |

Only two experiments pass statistical significance for Overall Winner. Context Quality has the strongest effect size of all experiments.

## Analysis

### Why It Works

1. **Structured prompt eliminates guesswork**: The original prompt said "here are data tables" and left the LLM to figure out that Backgrounds = global context, Reasoning Path = bridges, etc. The structured prompt makes this explicit, enabling better synthesis.

2. **Chunk reranking applies "lost in the middle" principle**: LLMs attend more to content at the beginning and end of context. Reranking puts the most query-relevant evidence first, maximizing the chance that the LLM grounds its answer in the best available evidence.

3. **Community filtering reduces topic confusion**: On queries that span multiple topics (e.g., entities from different documents), unfiltered community reports can include reports about unrelated topics. Filtering to top-5 by relevance keeps the global context focused.

4. **Zero-cost improvement**: Unlike hybrid retrieval (which adds embedding calls for BM25/PageRank), the structured prompt is completely free. Chunk reranking and community filtering add only ~2 small batch embedding calls per query.

### Diversity Gain

The strongest improvement is on Diversity (+28.4%, p<0.0001). This likely comes from the structured prompt encouraging the LLM to use all three context levels rather than defaulting to the most prominent one. By explicitly saying "use Backgrounds for framing, Reasoning Path for connections, Entity Details for specifics", the LLM naturally produces answers that incorporate multiple perspectives.

## Files

- Script: `eval/context_quality_eval.py`
- Summary: `eval/datasets/mix/mix_hi_ctxquality_k10_q130_summary.json`
- Metrics: `eval/datasets/mix/mix_hi_ctxquality_k10_q130_metrics.jsonl`
- Baseline answers: `eval/datasets/mix/mix_hi_ctxquality_k10_q130_answers_baseline.jsonl`
- Improved answers: `eval/datasets/mix/mix_hi_ctxquality_k10_q130_answers_improved.jsonl`
- Judge results: `eval/datasets/mix/mix_hi_ctxquality_k10_q130_judge_result_openai.jsonl`
