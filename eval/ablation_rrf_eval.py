"""Ablation study: which RRF signal contributes most?

Compares 4 configurations:
  1. Vector only (baseline)
  2. Vector + BM25
  3. Vector + PageRank
  4. Vector + BM25 + PageRank (full hybrid)

Each pair is judged by GPT-4o. The script runs all 6 pairwise comparisons.

Usage:
    cd eval/
    python ablation_rrf_eval.py --max-queries 130 --judge-openai
"""

import argparse
import json
import os
import sys
from pathlib import Path
from itertools import combinations

from tqdm import tqdm

from _common import config_value, dataset_dir, load_config

config = load_config()
os.environ["OPENAI_API_KEY"] = config_value(config, "openai", "api_key")
openai_base_url = config_value(config, "openai", "base_url", default="")
if openai_base_url and openai_base_url != "***":
    os.environ["OPENAI_BASE_URL"] = openai_base_url

from batch_eval import eval_oq_openai  # noqa: E402
from hirag import HiRAG, QueryParam  # noqa: E402
from hirag._utils import encode_string_by_tiktoken  # noqa: E402
from hirag._hybrid_retrieval import BM25Index, compute_pagerank, save_pagerank  # noqa: E402


def load_queries(input_path: Path, max_queries: int) -> list[str]:
    queries = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line)["input"])
    return queries[:max_queries]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_ratio(n, d):
    return n / d if d else 0.0


def count_tokens(text, model_name):
    if text is None:
        return 0
    return len(encode_string_by_tiktoken(text, model_name=model_name))


def ensure_hybrid_indexes(rag):
    graph = rag.chunk_entity_relation_graph._graph
    if graph.number_of_nodes() == 0:
        return
    if rag.bm25_index is None:
        entity_docs = {nid: dict(nd) if nd else {} for nid, nd in graph.nodes(data=True)}
        rag.bm25_index = BM25Index()
        rag.bm25_index.build(entity_docs)
        rag.bm25_index.save(rag._bm25_index_path)
    if rag.pagerank_scores is None:
        rag.pagerank_scores = compute_pagerank(graph, alpha=rag.pagerank_alpha)
        save_pagerank(rag.pagerank_scores, rag._pagerank_path)


CONFIGS = {
    "vector_only": {"bm25": False, "pr": False},
    "vector_bm25": {"bm25": True, "pr": False},
    "vector_pr": {"bm25": False, "pr": True},
    "vector_bm25_pr": {"bm25": True, "pr": True},
}


def query_with_config(rag, query, mode, cfg_name, saved_bm25, saved_pr):
    """Run a query with a specific ablation config."""
    cfg = CONFIGS[cfg_name]
    rag.bm25_index = saved_bm25 if cfg["bm25"] else None
    rag.pagerank_scores = saved_pr if cfg["pr"] else None
    use_hybrid = cfg["bm25"] or cfg["pr"]
    return rag.query(
        query=query,
        param=QueryParam(mode=mode, enable_hybrid_retrieval=use_hybrid),
    )


def summarize_judge(result_path, query_count, label_a, label_b):
    eval_path = result_path.with_name(result_path.stem + "_result_openai.jsonl")
    evaluations = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                evaluations.append(json.loads(line))
    criteria = ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]
    summary = {"judge_record_count": len(evaluations)}
    for criterion in criteria:
        a_wins = 0
        b_wins = 0
        for idx, item in enumerate(evaluations):
            winner = item[criterion]["Winner"]
            a_is_answer1 = idx < query_count
            if winner == "Answer 1":
                if a_is_answer1:
                    a_wins += 1
                else:
                    b_wins += 1
            elif winner == "Answer 2":
                if a_is_answer1:
                    b_wins += 1
                else:
                    a_wins += 1
        total = len(evaluations)
        summary[criterion] = {
            label_a: safe_ratio(a_wins, total),
            label_b: safe_ratio(b_wins, total),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Ablation: RRF signal contributions")
    parser.add_argument("-d", "--dataset", default="mix")
    parser.add_argument("-m", "--mode", default="hi")
    parser.add_argument("--working-dir", default="")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--judge-openai", action="store_true")
    args = parser.parse_args()

    dataset_root = dataset_dir(args.dataset)
    input_path = dataset_root / f"{args.dataset}.jsonl"
    working_dir = args.working_dir or str(dataset_root / "work_dir_openai_dedup_off_subset5")
    model_name = config_value(config, "openai", "model", "tiktoken_model_name", default="gpt-4o")

    rag = HiRAG(
        working_dir=working_dir,
        enable_hierachical_mode=args.mode != "naive",
        embedding_func_max_async=4,
        enable_naive_rag=args.mode == "naive",
        enable_hybrid_retrieval=True,
    )
    ensure_hybrid_indexes(rag)
    saved_bm25 = rag.bm25_index
    saved_pr = rag.pagerank_scores

    queries = load_queries(input_path, args.max_queries)
    print(f"Dataset: {args.dataset}, Mode: {args.mode}, Queries: {len(queries)}\n")

    # --- Generate answers for all 4 configs ---
    answers = {name: [] for name in CONFIGS}
    token_counts = {name: [] for name in CONFIGS}

    for query in tqdm(queries, desc="Generating answers"):
        for cfg_name in CONFIGS:
            answer = query_with_config(rag, query, args.mode, cfg_name, saved_bm25, saved_pr)
            answers[cfg_name].append({"query": query, "answer": answer})
            token_counts[cfg_name].append(count_tokens(answer, model_name))

    # restore
    rag.bm25_index = saved_bm25
    rag.pagerank_scores = saved_pr

    # --- Write answer files ---
    output_base = dataset_root / f"{args.dataset}_{args.mode}_ablation_q{args.max_queries}"
    query_path = Path(str(output_base) + "_queries.jsonl")
    write_jsonl(query_path, [{"input": q} for q in queries])

    for cfg_name, rows in answers.items():
        write_jsonl(Path(str(output_base) + f"_answers_{cfg_name}.jsonl"), rows)

    # --- Token summary ---
    summary = {"query_count": len(queries), "configs": {}}
    for cfg_name in CONFIGS:
        avg = sum(token_counts[cfg_name]) / max(len(token_counts[cfg_name]), 1)
        summary["configs"][cfg_name] = {"mean_answer_tokens": round(avg, 1)}
    print("\nToken summary:")
    for cfg_name, stats in summary["configs"].items():
        print(f"  {cfg_name:20s} mean_tokens={stats['mean_answer_tokens']}")

    # --- Judge: compare each config against vector_only baseline ---
    if args.judge_openai:
        baseline = "vector_only"
        comparisons = ["vector_bm25", "vector_pr", "vector_bm25_pr"]
        summary["judge_vs_baseline"] = {}

        for other in comparisons:
            print(f"\nJudging: {baseline} vs {other}")
            judge_path = Path(str(output_base) + f"_judge_{baseline}_vs_{other}.jsonl")
            eval_oq_openai(
                query_file=str(query_path),
                result1_file=str(output_base) + f"_answers_{baseline}.jsonl",
                result2_file=str(output_base) + f"_answers_{other}.jsonl",
                output_file_path=str(judge_path),
            )
            j = summarize_judge(judge_path, len(queries), baseline, other)
            summary["judge_vs_baseline"][f"{baseline}_vs_{other}"] = j
            print(f"  Results ({j['judge_record_count']} records):")
            for criterion in ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]:
                c = j[criterion]
                print(f"    {criterion:25s} {baseline}={c[baseline]:.3f}  {other}={c[other]:.3f}")

    summary_path = Path(str(output_base) + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
