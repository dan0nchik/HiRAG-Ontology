# Experiment Overview

## Research Context

We evaluate improvements to HiRAG (Hierarchical Retrieval-Augmented Generation) entity retrieval. The original HiRAG uses pure vector cosine similarity for entity retrieval. We propose replacing it with **Hybrid Retrieval via Reciprocal Rank Fusion (RRF)**, combining three complementary signals: vector similarity, BM25 lexical matching, and PageRank graph-structural importance.

## Dataset

- **UltraDomain Mix** (same as original HiRAG paper)
- 5 documents from the Mix subset, 625,948 tokens total
- Cached knowledge graph: **668 nodes, 587 edges**
- Pre-computed indexes: BM25 over entity descriptions, PageRank over graph

## Evaluation Protocol

- **LLM-as-judge**: GPT-4o evaluates answer pairs
- **4 criteria**: Comprehensiveness, Empowerment, Diversity, Overall Winner
- **Fairness**: each query evaluated twice with answer order swapped (A|B then B|A), producing 260 judge records for 130 queries
- **130 queries** from UltraDomain Mix benchmark

## Experiments Conducted

| # | Experiment | File | Queries | Cost est. |
|---|---|---|---|---|
| 1 | Hybrid Retrieval (RRF) vs Vector-only | `02_HYBRID_RETRIEVAL.md` | 130 | ~$30 |
| 2 | Ablation: RRF signal contributions | `03_ABLATION_RRF.md` | 130 | ~$90 |
| 3 | Token-normalized evaluation | `04_TOKEN_NORMALIZED.md` | 130×2 | ~$60 |
| 4 | MMR reranking | `05_MMR_RERANKING.md` | 130×2 | ~$60 |

Total: **~1,820 judge evaluations** across all experiments.

## Key Finding

Hybrid entity retrieval via RRF improves HiRAG answer quality by up to **+17.0%** (Overall Winner) while the improvement is not merely from increased context volume — at comparable token budgets it still achieves **+8.4%**.

## Branches and Reproducibility

- Code: `hybrid-retrieval` branch at `dan0nchik/HiRAG-Ontology`
- Experiment branch: `exp/mmr-reranking` (contains all eval scripts)
- Working directory: `eval/datasets/mix/work_dir_openai_dedup_off_subset5`
- All result files: `eval/datasets/mix/mix_hi_*_q130_*.json[l]`
