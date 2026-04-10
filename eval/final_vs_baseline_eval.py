"""Direct A/B evaluation: Final (Hybrid RRF k=10 + Context Quality) vs Vector-only baseline.

This is the end-to-end comparison of all cumulative improvements against the
original HiRAG vector-only retrieval.

Baseline: Vector-only retrieval (original HiRAG), top_k=20
Final:    Hybrid RRF (k=10) + structured prompt + chunk reranking + community filtering

Usage:
    cd eval/
    python final_vs_baseline_eval.py --max-queries 130 --judge-openai
"""

import argparse
import json
import os
import sys
from pathlib import Path

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


def safe_ratio(n: float, d: float) -> float:
    return n / d if d else 0.0


def count_tokens(text: str, model_name: str) -> int:
    if text is None:
        return 0
    return len(encode_string_by_tiktoken(text, model_name=model_name))


def ensure_hybrid_indexes(rag: HiRAG) -> None:
    graph = rag.chunk_entity_relation_graph._graph
    if graph.number_of_nodes() == 0:
        print("WARNING: Graph is empty.")
        return
    if rag.bm25_index is None:
        print(f"Building BM25 index over {graph.number_of_nodes()} entities...")
        entity_docs = {}
        for node_id, node_data in graph.nodes(data=True):
            entity_docs[node_id] = dict(node_data) if node_data else {}
        rag.bm25_index = BM25Index()
        rag.bm25_index.build(entity_docs)
        rag.bm25_index.save(rag._bm25_index_path)
    if rag.pagerank_scores is None:
        print(f"Computing PageRank for {graph.number_of_nodes()} nodes...")
        rag.pagerank_scores = compute_pagerank(graph, alpha=rag.pagerank_alpha)
        save_pagerank(rag.pagerank_scores, rag._pagerank_path)


def summarize_judge_results(result_path: Path, query_count: int) -> dict:
    eval_path = result_path.with_name(result_path.stem + "_result_openai.jsonl")
    evaluations = []
    with eval_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                evaluations.append(json.loads(line))

    criteria = ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]
    summary = {
        "judge_expected_count": query_count * 2,
        "judge_record_count": len(evaluations),
    }
    for criterion in criteria:
        baseline_wins = 0
        improved_wins = 0
        for index, item in enumerate(evaluations):
            winner = item[criterion]["Winner"]
            baseline_is_answer1 = index < query_count
            if winner == "Answer 1":
                if baseline_is_answer1:
                    baseline_wins += 1
                else:
                    improved_wins += 1
            elif winner == "Answer 2":
                if baseline_is_answer1:
                    improved_wins += 1
                else:
                    baseline_wins += 1
        total = len(evaluations)
        summary[criterion] = {
            "baseline": safe_ratio(baseline_wins, total),
            "improved": safe_ratio(improved_wins, total),
            "count": total,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B test: Final (all improvements) vs Vector-only baseline"
    )
    parser.add_argument("-d", "--dataset", type=str, default="mix")
    parser.add_argument("-m", "--mode", type=str, default="hi")
    parser.add_argument("--working-dir", type=str, default="")
    parser.add_argument("--max-queries", type=int, default=130)
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--judge-openai", action="store_true", help="Run GPT-4o judge")
    args = parser.parse_args()

    dataset_root = dataset_dir(args.dataset)
    input_path = dataset_root / f"{args.dataset}.jsonl"
    working_dir = args.working_dir or str(dataset_root / "work_dir_openai_dedup_off_subset5")
    output_prefix = args.output_prefix or str(
        dataset_root / f"{args.dataset}_{args.mode}_final_vs_baseline_q{args.max_queries}"
    )
    model_name = config_value(config, "openai", "model", "tiktoken_model_name", default="gpt-4o")

    print(f"Dataset:        {args.dataset}")
    print(f"Working dir:    {working_dir}")
    print(f"Mode:           {args.mode}")
    print(f"Max queries:    {args.max_queries}")
    print(f"Output prefix:  {output_prefix}")
    print()
    print("Baseline: Vector-only (original HiRAG), top_k=20")
    print("Final:    Hybrid RRF (k=10) + structured prompt + chunk reranking + community filtering")
    print()

    rag = HiRAG(
        working_dir=working_dir,
        enable_hierachical_mode=args.mode != "naive",
        embedding_func_max_async=4,
        enable_naive_rag=args.mode == "naive",
        enable_hybrid_retrieval=True,
    )
    ensure_hybrid_indexes(rag)

    queries = load_queries(input_path, args.max_queries)
    print(f"Loaded {len(queries)} queries\n")

    # --- QueryParam configs ---
    baseline_param = QueryParam(
        mode=args.mode,
        # Original vector-only: no hybrid, default top_k
        enable_hybrid_retrieval=False,
        top_k=20,
        use_structured_prompt=False,
        enable_chunk_reranking=False,
        enable_community_filtering=False,
    )
    final_param = QueryParam(
        mode=args.mode,
        # All improvements ON
        enable_hybrid_retrieval=True,
        top_k=10,
        use_structured_prompt=True,
        enable_chunk_reranking=True,
        enable_community_filtering=True,
        max_communities=5,
    )

    query_rows = []
    answer_rows_baseline = []
    answer_rows_final = []

    for index, query in enumerate(tqdm(queries, desc="Final vs Baseline"), start=1):
        # --- Baseline answer ---
        baseline_answer = rag.query(query=query, param=baseline_param)

        # --- Final answer ---
        final_answer = rag.query(query=query, param=final_param)

        baseline_tokens = count_tokens(baseline_answer, model_name)
        final_tokens = count_tokens(final_answer, model_name)

        row = {
            "index": index,
            "query": query,
            "baseline_answer_tokens": baseline_tokens,
            "final_answer_tokens": final_tokens,
        }
        query_rows.append(row)
        answer_rows_baseline.append({"query": query, "answer": baseline_answer})
        answer_rows_final.append({"query": query, "answer": final_answer})

    # --- Write outputs ---
    metrics_path = Path(output_prefix + "_metrics.jsonl")
    query_path = Path(output_prefix + "_queries.jsonl")
    baseline_path = Path(output_prefix + "_answers_baseline.jsonl")
    final_path = Path(output_prefix + "_answers_final.jsonl")

    write_jsonl(metrics_path, query_rows)
    write_jsonl(query_path, [{"input": row["query"]} for row in query_rows])
    write_jsonl(baseline_path, answer_rows_baseline)
    write_jsonl(final_path, answer_rows_final)

    summary = {
        "experiment": "final_vs_vector_only_baseline",
        "baseline": "vector-only (original HiRAG), top_k=20",
        "final": "hybrid RRF k=10 + structured_prompt + chunk_reranking + community_filtering",
        "query_count": len(queries),
    }

    if args.judge_openai:
        judge_path = Path(output_prefix + "_judge.jsonl")
        eval_oq_openai(
            query_file=str(query_path),
            result1_file=str(baseline_path),
            result2_file=str(final_path),
            output_file_path=str(judge_path),
        )
        judge_result_path = judge_path.with_name(judge_path.stem + "_result_openai.jsonl")
        summary["judge_openai"] = summarize_judge_results(judge_path, query_count=len(queries))

    summary_path = Path(output_prefix + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Queries: {len(queries)}")
    if "judge_openai" in summary:
        j = summary["judge_openai"]
        print(f"\nJudge results ({j.get('judge_record_count', 0)} records):")
        for criterion in ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]:
            if criterion in j:
                c = j[criterion]
                print(f"  {criterion:25s} baseline={c['baseline']:.3f}  final={c['improved']:.3f}")
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
