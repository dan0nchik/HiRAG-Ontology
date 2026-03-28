"""A/B evaluation: Query-Aware Bridge Path strategies vs shortest-path baseline.

Uses an existing cached graph — no reindexing required.
Edge embeddings are built on first run from the cached graph and persisted.

Usage:
    cd eval/
    python bridge_strategy_eval.py --max-queries 10
    python bridge_strategy_eval.py --max-queries 10 --judge-openai
    python bridge_strategy_eval.py --strategy dijkstra_qa --max-queries 50 --judge-openai
"""

import argparse
import asyncio
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
from hirag._bridge_paths import (  # noqa: E402
    compute_edge_embeddings,
    save_edge_embeddings,
    load_edge_embeddings,
    STRATEGIES,
)


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


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def count_tokens(text: str, model_name: str) -> int:
    if text is None:
        return 0
    return len(encode_string_by_tiktoken(text, model_name=model_name))


def ensure_hybrid_indexes(rag: HiRAG) -> None:
    """Build BM25 + PageRank indexes from cached graph if not already present."""
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


def ensure_edge_embeddings(rag: HiRAG) -> None:
    """Build edge embeddings from cached graph if not already present."""
    if rag.edge_embeddings is not None:
        return
    graph = rag.chunk_entity_relation_graph._graph
    edge_count = graph.number_of_edges()
    if edge_count == 0:
        print("WARNING: Graph has no edges.")
        return
    print(f"Computing edge embeddings for {edge_count} edges...")
    loop = asyncio.get_event_loop()
    rag.edge_embeddings = loop.run_until_complete(
        compute_edge_embeddings(graph, rag.embedding_func)
    )
    save_edge_embeddings(rag.edge_embeddings, rag._edge_embeddings_path)
    print(f"  -> saved {len(rag.edge_embeddings)} edge embeddings to {rag._edge_embeddings_path}")


def summarize_judge_results(result_path: Path, query_count: int, label_a: str, label_b: str) -> dict:
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
        a_wins = 0
        b_wins = 0
        for index, item in enumerate(evaluations):
            winner = item[criterion]["Winner"]
            a_is_answer1 = index < query_count
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
            "count": total,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test: bridge path strategies")
    parser.add_argument("-d", "--dataset", type=str, default="mix")
    parser.add_argument("-m", "--mode", type=str, default="hi")
    parser.add_argument("--working-dir", type=str, default="")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument(
        "--strategy", type=str, default="dijkstra_qa",
        help=f"Bridge strategy to test. Options: {', '.join(sorted(STRATEGIES))}",
    )
    parser.add_argument(
        "--pruning-threshold", type=float, default=0.1,
        help="Min edge relevance for pruning (0 = no pruning)",
    )
    parser.add_argument("--skip-answers", action="store_true")
    parser.add_argument("--judge-openai", action="store_true")
    parser.add_argument("--enable-hybrid", action="store_true", default=True,
                        help="Enable hybrid retrieval for both baseline and test (default: True)")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Disable hybrid retrieval for both baseline and test")
    args = parser.parse_args()

    enable_hybrid = not args.no_hybrid

    dataset_root = dataset_dir(args.dataset)
    input_path = dataset_root / f"{args.dataset}.jsonl"
    working_dir = args.working_dir or str(dataset_root / "work_dir_openai_dedup_off_subset5")
    strategy_label = args.strategy
    if args.pruning_threshold > 0:
        strategy_label += "_pruned"
    output_prefix = args.output_prefix or str(
        dataset_root / f"{args.dataset}_{args.mode}_bridge_{strategy_label}_q{args.max_queries}"
    )
    model_name = config_value(config, "openai", "model", "tiktoken_model_name", default="gpt-4o")

    print(f"Dataset: {args.dataset}")
    print(f"Working dir: {working_dir}")
    print(f"Mode: {args.mode}")
    print(f"Bridge strategy: {args.strategy} (baseline: shortest)")
    print(f"Pruning threshold: {args.pruning_threshold}")
    print(f"Hybrid retrieval: {'ON' if enable_hybrid else 'OFF'}")
    print(f"Max queries: {args.max_queries}")
    print()

    rag = HiRAG(
        working_dir=working_dir,
        enable_hierachical_mode=args.mode != "naive",
        embedding_func_max_async=4,
        enable_naive_rag=args.mode == "naive",
        enable_hybrid_retrieval=enable_hybrid,
        enable_query_aware_bridge=True,
    )

    if enable_hybrid:
        ensure_hybrid_indexes(rag)
    ensure_edge_embeddings(rag)

    queries = load_queries(input_path, args.max_queries)
    print(f"Loaded {len(queries)} queries\n")

    query_rows = []
    answer_rows_baseline = []
    answer_rows_strategy = []

    for index, query in enumerate(tqdm(queries, desc=f"A/B bridge eval ({args.strategy})"), start=1):
        # --- Baseline: shortest path ---
        baseline_context = rag.query(
            query=query,
            param=QueryParam(
                mode=args.mode,
                enable_hybrid_retrieval=enable_hybrid,
                bridge_strategy="shortest",
                bridge_min_edge_relevance=0,
                only_need_context=True,
            ),
        )
        # --- Test: configured strategy ---
        strategy_context = rag.query(
            query=query,
            param=QueryParam(
                mode=args.mode,
                enable_hybrid_retrieval=enable_hybrid,
                bridge_strategy=args.strategy,
                bridge_min_edge_relevance=args.pruning_threshold,
                only_need_context=True,
            ),
        )

        baseline_tokens = count_tokens(baseline_context, model_name)
        strategy_tokens = count_tokens(strategy_context, model_name)

        row = {
            "index": index,
            "query": query,
            "baseline_tokens": baseline_tokens,
            "strategy_tokens": strategy_tokens,
            "token_delta": strategy_tokens - baseline_tokens,
            "contexts_identical": (baseline_context == strategy_context),
        }
        query_rows.append(row)

        if not args.skip_answers:
            baseline_answer = rag.query(
                query=query,
                param=QueryParam(
                    mode=args.mode,
                    enable_hybrid_retrieval=enable_hybrid,
                    bridge_strategy="shortest",
                    bridge_min_edge_relevance=0,
                ),
            )
            strategy_answer = rag.query(
                query=query,
                param=QueryParam(
                    mode=args.mode,
                    enable_hybrid_retrieval=enable_hybrid,
                    bridge_strategy=args.strategy,
                    bridge_min_edge_relevance=args.pruning_threshold,
                ),
            )
            answer_rows_baseline.append({"query": query, "answer": baseline_answer})
            answer_rows_strategy.append({"query": query, "answer": strategy_answer})

    # --- Aggregate ---
    total = len(query_rows)
    identical_count = sum(1 for r in query_rows if r["contexts_identical"])
    changed_count = total - identical_count
    mean_baseline_tokens = sum(r["baseline_tokens"] for r in query_rows) / max(total, 1)
    mean_strategy_tokens = sum(r["strategy_tokens"] for r in query_rows) / max(total, 1)
    mean_token_delta = sum(r["token_delta"] for r in query_rows) / max(total, 1)

    summary = {
        "query_count": total,
        "baseline_strategy": "shortest",
        "test_strategy": strategy_label,
        "hybrid_retrieval": enable_hybrid,
        "pruning_threshold": args.pruning_threshold,
        "contexts_identical": identical_count,
        "contexts_changed": changed_count,
        "pct_changed": safe_ratio(changed_count, total) * 100,
        "mean_baseline_tokens": round(mean_baseline_tokens, 1),
        "mean_strategy_tokens": round(mean_strategy_tokens, 1),
        "mean_token_delta": round(mean_token_delta, 1),
    }

    # --- Write outputs ---
    metrics_path = Path(output_prefix + "_metrics.jsonl")
    summary_path = Path(output_prefix + "_summary.json")
    query_path = Path(output_prefix + "_queries.jsonl")
    write_jsonl(metrics_path, query_rows)
    write_jsonl(query_path, [{"input": row["query"]} for row in query_rows])
    outputs = {"metrics": str(metrics_path), "queries": str(query_path)}

    if not args.skip_answers:
        baseline_path = Path(output_prefix + "_answers_baseline.jsonl")
        strategy_path = Path(output_prefix + "_answers_strategy.jsonl")
        write_jsonl(baseline_path, answer_rows_baseline)
        write_jsonl(strategy_path, answer_rows_strategy)
        outputs["answers_baseline"] = str(baseline_path)
        outputs["answers_strategy"] = str(strategy_path)

        if args.judge_openai:
            judge_path = Path(output_prefix + "_judge.jsonl")
            eval_oq_openai(
                query_file=str(query_path),
                result1_file=str(baseline_path),
                result2_file=str(strategy_path),
                output_file_path=str(judge_path),
            )
            outputs["judge"] = str(judge_path.with_name(judge_path.stem + "_result_openai.jsonl"))
            summary["judge_openai"] = summarize_judge_results(
                judge_path, query_count=total,
                label_a="shortest", label_b=strategy_label,
            )

    summary["outputs"] = outputs
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline:             shortest")
    print(f"Test strategy:        {strategy_label}")
    print(f"Queries:              {total}")
    print(f"Contexts changed:     {changed_count}/{total} ({summary['pct_changed']:.1f}%)")
    print(f"Mean baseline tokens: {summary['mean_baseline_tokens']}")
    print(f"Mean strategy tokens: {summary['mean_strategy_tokens']}")
    print(f"Mean token delta:     {summary['mean_token_delta']}")
    if "judge_openai" in summary:
        j = summary["judge_openai"]
        print(f"\nJudge results ({j.get('judge_record_count', 0)} records):")
        for criterion in ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]:
            if criterion in j:
                c = j[criterion]
                print(f"  {criterion:25s} shortest={c['shortest']:.3f}  {strategy_label}={c[strategy_label]:.3f}")
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
