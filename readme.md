# HiRAG: Hierarchical Retrieval-Augmented Generation

> Fork of [hhy-huang/HiRAG](https://github.com/hhy-huang/HiRAG) with experimental improvements to entity retrieval.

Based on the paper [Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/abs/2503.10150) (EMNLP 2025 Findings).

---

## Our Improvements

This fork implements and evaluates a series of improvements to HiRAG's retrieval pipeline. All experiments were conducted on **130 queries** from UltraDomain Mix, judged by GPT-4o (260 judge records per comparison with answer order swapping).

### Hybrid Entity Retrieval via RRF

Replaced single-signal vector search with multi-signal fusion via Reciprocal Rank Fusion (RRF), combining:
- **Vector similarity** (semantic relevance)
- **BM25** (lexical precision)
- **PageRank** (graph-structural importance)

| Criterion | Vector-only | Hybrid RRF (k=10) | Delta |
|---|---|---|---|
| Comprehensiveness | 0.435 | **0.565** | +13.0% |
| Empowerment | 0.388 | **0.612** | +22.4% |
| Diversity | 0.446 | **0.554** | +10.8% |
| **Overall Winner** | 0.415 | **0.585** | **+17.0%** |

**p=0.0075** (statistically significant at p<0.01). Zero additional LLM calls — indexes computed once at build time.

Key finding: optimal quality at **k=10**, not k=20. Fewer but better-selected entities reduce noise. At near-equal token budget (k=7, 1.14x tokens) the improvement is still **+8.4%**.

### Ablation: RRF Signal Contributions

| Configuration | Overall | Delta |
|---|---|---|
| Vector only (baseline) | 0.500 | -- |
| + BM25 | 0.515 | +3.1% |
| + PageRank | 0.508 | +1.5% |
| + BM25 + PageRank | **0.550** | **+10.0%** |

Synergy: combined improvement (+10.0%) exceeds sum of individual signals (+4.6%).

### Negative Results

- **MMR reranking**: -2.3% to -4.6% on Overall. Forcing diversity at entity selection hurts comprehensiveness. The bottleneck is entity identification quality, not diversity.

### Summary: All Experiments

| Experiment | Overall Win Rate | p-value | Significant? | Cohen's h |
|---|---|---|---|---|
| Hybrid RRF k=20 vs Vector-only | 0.546 | 0.1536 | No | +0.092 |
| Hybrid RRF k=10 vs Vector-only | **0.585** | **0.0075** | **Yes (p<0.01)** | +0.170 |
| Hybrid RRF k=7 vs Vector-only | 0.542 | 0.1927 | No | +0.085 |
| MMR lambda=0.7 vs Hybrid | 0.477 | 0.4952 | No | -0.046 |
| MMR lambda=0.9 vs Hybrid | 0.488 | 0.7566 | No | -0.023 |

### Detailed Experiment Reports

- [Experiment Overview](paper_experiments/01_EXPERIMENT_OVERVIEW.md)
- [Hybrid Retrieval (RRF)](paper_experiments/02_HYBRID_RETRIEVAL.md)
- [Ablation: RRF Signal Contributions](paper_experiments/03_ABLATION_RRF.md)
- [Token-Normalized Evaluation](paper_experiments/04_TOKEN_NORMALIZED.md)
- [MMR Reranking (Negative Result)](paper_experiments/05_MMR_RERANKING.md)
- [Consolidated Tables](paper_experiments/06_CONSOLIDATED_TABLES.md)
- [Paper Narrative](paper_experiments/07_PAPER_NARRATIVE.md)
- [Statistical Significance Tests](paper_experiments/08_SIGNIFICANCE_TESTS.md)
---

## Install

```bash
cd HiRAG
pip install -e .
```

## Quick Start

```python
graph_func = HiRAG(
    working_dir="./your_work_dir",
    enable_llm_cache=True,
    enable_hierachical_mode=True, 
    embedding_batch_num=6,
    embedding_func_max_async=8,
    enable_naive_rag=True
    )
# indexing
with open("path_to_your_context.txt", "r") as f:
    graph_func.insert(f.read())
# retrieval & generation
print(graph_func.query("Your question?", param=QueryParam(mode="hi")))
```

To use hybrid retrieval:

```python
from hirag import QueryParam

param = QueryParam(
    mode="hi",
    enable_hybrid_retrieval=True,
    top_k=10,
)
print(graph_func.query("Your question?", param=param))
```

API keys and LLM configuration: `./config.yaml`. Examples for DeepSeek, ChatGLM, OpenAI: see `./hi_Search_*.py`.

## Evaluation

```shell
cd ./HiRAG/eval
```

1. Extract context:
```shell
python extract_context.py -i ./datasets/mix -o ./datasets/mix
```

2. Insert context to Graph Database:
```shell
python insert_context_deepseek.py
```

3. Test:
```shell
python test_deepseek.py -d mix -m hi        # HiRAG
python test_deepseek.py -d mix -m naive      # Naive RAG
python test_deepseek.py -d mix -m hi_nobridge  # without bridge
```

4. Evaluate:
```shell
python batch_eval.py -m request -api openai
python batch_eval.py -m result -api openai
```

5. Statistical significance tests:
```shell
python significance_tests.py --format markdown
```

## Acknowledgement

- Original paper: [HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/abs/2503.10150) by Huang et al.
- [nano-graphrag](https://github.com/gusye1234/nano-graphrag): simple, hackable GraphRAG implementation
- [RAPTOR](https://github.com/parthsarthi03/raptor): recursive tree structure for retrieval-augmented LMs

## Citation

```bibtex
@article{huang2025retrieval,
  title={Retrieval-Augmented Generation with Hierarchical Knowledge},
  author={Huang, Haoyu and Huang, Yongfeng and Yang, Junjie and Pan, Zhenyu and Chen, Yongqiang and Ma, Kaili and Chen, Hongzhi and Cheng, James},
  journal={arXiv preprint arXiv:2503.10150},
  year={2025}
}
```
