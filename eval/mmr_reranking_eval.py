"""A/B evaluation: MMR reranking vs standard top-k selection.

Both variants use hybrid retrieval (BM25 + Vector + PageRank via RRF).
The only difference is how top-k entities are selected from the candidate pool:
  - Baseline: simple truncation (first top_k by RRF score)
  - MMR: Maximal Marginal Relevance (balances relevance and diversity)

Usage:
    cd eval/
    python mmr_reranking_eval.py --max-queries 10
    python mmr_reranking_eval.py --max-queries 130 --judge-openai
    python mmr_reranking_eval.py --max-queries 10 --mmr-lambda 0.5
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
        mmr_wins = 0
        for index, item in enumerate(evaluations):
            winner = item[criterion]["Winner"]
            baseline_is_answer1 = index < query_count
            if winner == "Answer 1":
                if baseline_is_answer1:
                    baseline_wins += 1
                else:
                    mmr_wins += 1
            elif winner == "Answer 2":
                if baseline_is_answer1:
                    mmr_wins += 1
                else:
                    baseline_wins += 1
        total = len(evaluations)
        summary[criterion] = {
            "baseline_hybrid": safe_ratio(baseline_wins, total),
            "mmr_hybrid": safe_ratio(mmr_wins, total),
            "count": total,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test: MMR reranking vs standard top-k")
    parser.add_argument("-d", "--dataset", type=str, default="mix")
    parser.add_argument("-m", "--mode", type=str, default="hi")
    parser.add_argument("--working-dir", type=str, default="")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--skip-answers", action="store_true")
    parser.add_argument("--judge-openai", action="store_true")
    args = parser.parse_args()

    dataset_root = dataset_dir(args.dataset)
    input_path = dataset_root / f"{args.dataset}.jsonl"
    working_dir = args.working_dir or str(dataset_root / "work_dir_openai_dedup_off_subset5")
    lam_str = str(args.mmr_lambda).replace(".", "")
    output_prefix = args.output_prefix or str(
        dataset_root / f"{args.dataset}_{args.mode}_mmr_lam{lam_str}_q{args.max_queries}"
    )
    model_name = config_value(config, "openai", "model", "tiktoken_model_name", default="gpt-4o")

    print(f"Dataset: {args.dataset}")
    print(f"Working dir: {working_dir}")
    print(f"Mode: {args.mode}")
    print(f"MMR lambda: {args.mmr_lambda}")
    print(f"Max queries: {args.max_queries}")
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

    query_rows = []
    answer_rows_baseline = []
    answer_rows_mmr = []

    for index, query in enumerate(tqdm(queries, desc="A/B MMR eval"), start=1):
        # --- Baseline: hybrid retrieval, no MMR ---
        baseline_context = rag.query(
            query=query,
            param=QueryParam(mode=args.mode, enable_hybrid_retrieval=True, mmr_lambda=0.0, only_need_context=True),
        )
        # --- MMR: hybrid retrieval + MMR reranking ---
        mmr_context = rag.query(
            query=query,
            param=QueryParam(mode=args.mode, enable_hybrid_retrieval=True, mmr_lambda=args.mmr_lambda, only_need_context=True),
        )

        baseline_tokens = count_tokens(baseline_context, model_name)
        mmr_tokens = count_tokens(mmr_context, model_name)

        row = {
            "index": index,
            "query": query,
            "baseline_tokens": baseline_tokens,
            "mmr_tokens": mmr_tokens,
            "token_delta": mmr_tokens - baseline_tokens,
            "contexts_identical": (baseline_context == mmr_context),
        }
        query_rows.append(row)

        if not args.skip_answers:
            baseline_answer = rag.query(
                query=query,
                param=QueryParam(mode=args.mode, enable_hybrid_retrieval=True, mmr_lambda=0.0),
            )
            mmr_answer = rag.query(
                query=query,
                param=QueryParam(mode=args.mode, enable_hybrid_retrieval=True, mmr_lambda=args.mmr_lambda),
            )
            answer_rows_baseline.append({"query": query, "answer": baseline_answer})
            answer_rows_mmr.append({"query": query, "answer": mmr_answer})

    # --- Aggregate ---
    total = len(query_rows)
    identical_count = sum(1 for r in query_rows if r["contexts_identical"])
    changed_count = total - identical_count
    mean_baseline = sum(r["baseline_tokens"] for r in query_rows) / max(total, 1)
    mean_mmr = sum(r["mmr_tokens"] for r in query_rows) / max(total, 1)
    mean_delta = sum(r["token_delta"] for r in query_rows) / max(total, 1)

    summary = {
        "query_count": total,
        "mmr_lambda": args.mmr_lambda,
        "contexts_identical": identical_count,
        "contexts_changed": changed_count,
        "pct_changed": safe_ratio(changed_count, total) * 100,
        "mean_baseline_tokens": round(mean_baseline, 1),
        "mean_mmr_tokens": round(mean_mmr, 1),
        "mean_token_delta": round(mean_delta, 1),
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
        mmr_path = Path(output_prefix + "_answers_mmr.jsonl")
        write_jsonl(baseline_path, answer_rows_baseline)
        write_jsonl(mmr_path, answer_rows_mmr)
        outputs["answers_baseline"] = str(baseline_path)
        outputs["answers_mmr"] = str(mmr_path)

        if args.judge_openai:
            judge_path = Path(output_prefix + "_judge.jsonl")
            eval_oq_openai(
                query_file=str(query_path),
                result1_file=str(baseline_path),
                result2_file=str(mmr_path),
                output_file_path=str(judge_path),
            )
            outputs["judge"] = str(judge_path.with_name(judge_path.stem + "_result_openai.jsonl"))
            summary["judge_openai"] = summarize_judge_results(judge_path, query_count=total)

    summary["outputs"] = outputs
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Queries:              {total}")
    print(f"MMR lambda:           {args.mmr_lambda}")
    print(f"Contexts changed:     {changed_count}/{total} ({summary['pct_changed']:.1f}%)")
    print(f"Mean baseline tokens: {summary['mean_baseline_tokens']}")
    print(f"Mean MMR tokens:      {summary['mean_mmr_tokens']}")
    print(f"Mean token delta:     {summary['mean_token_delta']}")
    if "judge_openai" in summary:
        j = summary["judge_openai"]
        print(f"\nJudge results ({j.get('judge_record_count', 0)} valid records):")
        for criterion in ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]:
            if criterion in j:
                c = j[criterion]
                print(f"  {criterion:25s} baseline={c['baseline_hybrid']:.3f}  mmr={c['mmr_hybrid']:.3f}")
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
