"""Token-normalized evaluation: same token budget, better quality?

Runs hybrid retrieval with reduced top_k so that the context size matches
the baseline (vector-only) token count. If quality is still better,
this proves the improvement isn't just from using more context.

Usage:
    cd eval/
    python token_normalized_eval.py --max-queries 130 --judge-openai
    python token_normalized_eval.py --max-queries 10 --hybrid-top-k 10
"""

import argparse
import json
import os
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


def load_queries(input_path, max_queries):
    queries = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line)["input"])
    return queries[:max_queries]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def summarize_judge(result_path, query_count):
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
        baseline_wins = 0
        hybrid_wins = 0
        for idx, item in enumerate(evaluations):
            winner = item[criterion]["Winner"]
            baseline_is_1 = idx < query_count
            if winner == "Answer 1":
                if baseline_is_1:
                    baseline_wins += 1
                else:
                    hybrid_wins += 1
            elif winner == "Answer 2":
                if baseline_is_1:
                    hybrid_wins += 1
                else:
                    baseline_wins += 1
        total = len(evaluations)
        summary[criterion] = {
            "baseline": safe_ratio(baseline_wins, total),
            "hybrid_budget": safe_ratio(hybrid_wins, total),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Token-normalized A/B: hybrid at same token budget vs baseline")
    parser.add_argument("-d", "--dataset", default="mix")
    parser.add_argument("-m", "--mode", default="hi")
    parser.add_argument("--working-dir", default="")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--baseline-top-k", type=int, default=20, help="top_k for baseline (vector-only)")
    parser.add_argument("--hybrid-top-k", type=int, default=10, help="top_k for hybrid (reduced budget)")
    parser.add_argument("--judge-openai", action="store_true")
    args = parser.parse_args()

    dataset_root = dataset_dir(args.dataset)
    input_path = dataset_root / f"{args.dataset}.jsonl"
    working_dir = args.working_dir or str(dataset_root / "work_dir_openai_dedup_off_subset5")
    model_name = config_value(config, "openai", "model", "tiktoken_model_name", default="gpt-4o")
    output_prefix = str(
        dataset_root / f"{args.dataset}_{args.mode}_tokennorm_bk{args.baseline_top_k}_hk{args.hybrid_top_k}_q{args.max_queries}"
    )

    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Baseline top_k: {args.baseline_top_k} (vector-only)")
    print(f"Hybrid top_k: {args.hybrid_top_k} (RRF, reduced budget)")
    print(f"Max queries: {args.max_queries}\n")

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

    # --- Phase 1: context-only pass to calibrate tokens ---
    print("Phase 1: Measuring token counts...")
    baseline_tokens_list = []
    hybrid_tokens_list = []

    for query in tqdm(queries[:min(20, len(queries))], desc="Calibrating"):
        bc = rag.query(query, param=QueryParam(
            mode=args.mode, enable_hybrid_retrieval=False,
            top_k=args.baseline_top_k, only_need_context=True,
        ))
        hc = rag.query(query, param=QueryParam(
            mode=args.mode, enable_hybrid_retrieval=True,
            top_k=args.hybrid_top_k, only_need_context=True,
        ))
        baseline_tokens_list.append(count_tokens(bc, model_name))
        hybrid_tokens_list.append(count_tokens(hc, model_name))

    avg_baseline = sum(baseline_tokens_list) / len(baseline_tokens_list)
    avg_hybrid = sum(hybrid_tokens_list) / len(hybrid_tokens_list)
    print(f"\nCalibration (first {len(baseline_tokens_list)} queries):")
    print(f"  Baseline (vector, top_k={args.baseline_top_k}): {avg_baseline:.0f} tokens")
    print(f"  Hybrid (RRF, top_k={args.hybrid_top_k}):   {avg_hybrid:.0f} tokens")
    print(f"  Ratio: {avg_hybrid / avg_baseline:.2f}x")
    print()

    # --- Phase 2: full A/B with answers ---
    query_rows = []
    answer_baseline = []
    answer_hybrid = []

    for idx, query in enumerate(tqdm(queries, desc="A/B token-norm eval"), start=1):
        # baseline: vector-only, standard top_k
        ctx_b = rag.query(query, param=QueryParam(
            mode=args.mode, enable_hybrid_retrieval=False,
            top_k=args.baseline_top_k, only_need_context=True,
        ))
        # hybrid: RRF, reduced top_k
        ctx_h = rag.query(query, param=QueryParam(
            mode=args.mode, enable_hybrid_retrieval=True,
            top_k=args.hybrid_top_k, only_need_context=True,
        ))

        bt = count_tokens(ctx_b, model_name)
        ht = count_tokens(ctx_h, model_name)

        query_rows.append({
            "index": idx, "query": query,
            "baseline_tokens": bt, "hybrid_tokens": ht,
            "token_delta": ht - bt,
        })

        # answers
        ans_b = rag.query(query, param=QueryParam(
            mode=args.mode, enable_hybrid_retrieval=False, top_k=args.baseline_top_k,
        ))
        ans_h = rag.query(query, param=QueryParam(
            mode=args.mode, enable_hybrid_retrieval=True, top_k=args.hybrid_top_k,
        ))
        answer_baseline.append({"query": query, "answer": ans_b})
        answer_hybrid.append({"query": query, "answer": ans_h})

    # --- Aggregate ---
    total = len(query_rows)
    mean_bt = sum(r["baseline_tokens"] for r in query_rows) / total
    mean_ht = sum(r["hybrid_tokens"] for r in query_rows) / total

    summary = {
        "query_count": total,
        "baseline_top_k": args.baseline_top_k,
        "hybrid_top_k": args.hybrid_top_k,
        "mean_baseline_tokens": round(mean_bt, 1),
        "mean_hybrid_tokens": round(mean_ht, 1),
        "mean_token_delta": round(mean_ht - mean_bt, 1),
        "token_ratio": round(mean_ht / mean_bt, 3) if mean_bt > 0 else 0,
    }

    # --- Write ---
    metrics_path = Path(output_prefix + "_metrics.jsonl")
    summary_path = Path(output_prefix + "_summary.json")
    query_path = Path(output_prefix + "_queries.jsonl")
    baseline_path = Path(output_prefix + "_answers_baseline.jsonl")
    hybrid_path = Path(output_prefix + "_answers_hybrid.jsonl")

    write_jsonl(metrics_path, query_rows)
    write_jsonl(query_path, [{"input": r["query"]} for r in query_rows])
    write_jsonl(baseline_path, answer_baseline)
    write_jsonl(hybrid_path, answer_hybrid)

    if args.judge_openai:
        judge_path = Path(output_prefix + "_judge.jsonl")
        eval_oq_openai(
            query_file=str(query_path),
            result1_file=str(baseline_path),
            result2_file=str(hybrid_path),
            output_file_path=str(judge_path),
        )
        summary["judge_openai"] = summarize_judge(judge_path, total)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Queries:              {total}")
    print(f"Baseline:             vector-only, top_k={args.baseline_top_k}")
    print(f"Hybrid:               RRF, top_k={args.hybrid_top_k}")
    print(f"Mean baseline tokens: {summary['mean_baseline_tokens']}")
    print(f"Mean hybrid tokens:   {summary['mean_hybrid_tokens']}")
    print(f"Token ratio:          {summary['token_ratio']}x")
    if "judge_openai" in summary:
        j = summary["judge_openai"]
        print(f"\nJudge ({j['judge_record_count']} records):")
        for c in ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]:
            if c in j:
                v = j[c]
                print(f"  {c:25s} baseline={v['baseline']:.3f}  hybrid_budget={v['hybrid_budget']:.3f}")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
