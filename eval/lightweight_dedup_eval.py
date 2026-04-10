import argparse
import csv
import io
import json
import os
import re
from hashlib import md5
from pathlib import Path

from tqdm import tqdm

from _common import config_value, dataset_dir, load_config
from batch_eval import eval_oq_openai
from hirag import HiRAG, QueryParam
from hirag._utils import encode_string_by_tiktoken


config = load_config()
os.environ["OPENAI_API_KEY"] = config_value(config, "openai", "api_key")
openai_base_url = config_value(config, "openai", "base_url", default="")
if openai_base_url and openai_base_url != "***":
    os.environ["OPENAI_BASE_URL"] = openai_base_url


SECTION_HEADERS = {
    "backgrounds": [
        "-----Backgrounds-----",
    ],
    "entities": [
        "-----Entities-----",
        "-----Detail Entity Information-----",
    ],
    "relations": [
        "-----Relations-----",
        "-----Reasoning Path-----",
    ],
    "sources": [
        "-----Sources-----",
        "-----Source Documents-----",
    ],
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fingerprint_text(text: str) -> str:
    return md5(normalize_text(text).lower().encode("utf-8")).hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def extract_csv_section_text(context: str, section_name: str) -> str:
    for header in SECTION_HEADERS[section_name]:
        pattern = rf"{re.escape(header)}\s*```csv\s*(.*?)\s*```"
        match = re.search(pattern, context, re.DOTALL)
        if match is None:
            continue
        content = match.group(1).strip()
        if not content:
            return ""
        return content
    return ""


def parse_csv_section_rows(content: str) -> list[list[str]]:
    if not content:
        return []
    header_line, _, body = content.partition("\n")
    rows = [next(csv.reader(io.StringIO(header_line)))]
    if not body.strip():
        return rows

    row_chunks = [
        chunk for chunk in re.split(r"(?m)(?=^\d+,)", body) if chunk.strip()
    ]
    for chunk in row_chunks:
        parsed_rows = [row for row in csv.reader(io.StringIO(chunk)) if row]
        if parsed_rows:
            rows.append(parsed_rows[0])
    return rows


def parse_hierarchical_context(context: str) -> dict:
    background_rows = parse_csv_section_rows(extract_csv_section_text(context, "backgrounds"))
    entity_rows = parse_csv_section_rows(extract_csv_section_text(context, "entities"))
    relation_rows = parse_csv_section_rows(extract_csv_section_text(context, "relations"))
    source_rows = parse_csv_section_rows(extract_csv_section_text(context, "sources"))

    backgrounds = background_rows[1:] if len(background_rows) > 1 else []
    entities = entity_rows[1:] if len(entity_rows) > 1 else []
    relations = relation_rows[1:] if len(relation_rows) > 1 else []
    sources = source_rows[1:] if len(source_rows) > 1 else []

    entity_names = [row[1] for row in entities if len(row) > 1]
    relation_pairs = [
        tuple(sorted((row[1], row[2])))
        for row in relations
        if len(row) > 2
    ]
    relation_entries = [
        (
            tuple(sorted((row[1], row[2]))),
            fingerprint_text(row[3] if len(row) > 3 else ""),
        )
        for row in relations
        if len(row) > 2
    ]
    source_contents = [row[1] for row in sources if len(row) > 1]
    source_fingerprints = [fingerprint_text(content) for content in source_contents]

    return {
        "format": "hierarchical",
        "background_rows": len(backgrounds),
        "entity_rows": len(entities),
        "unique_entities": len({normalize_text(name).lower() for name in entity_names if name}),
        "relation_rows": len(relations),
        "unique_relation_pairs": len(set(relation_pairs)),
        "duplicate_relation_pairs": len(relations) - len(set(relation_pairs)),
        "unique_relation_entries": len(set(relation_entries)),
        "duplicate_relation_entries": len(relations) - len(set(relation_entries)),
        "source_rows": len(sources),
        "unique_source_rows": len(set(source_fingerprints)),
        "duplicate_source_rows": len(sources) - len(set(source_fingerprints)),
    }


def parse_naive_context(context: str) -> dict:
    chunks = [chunk.strip() for chunk in context.split("--New Chunk--") if chunk.strip()]
    fingerprints = [fingerprint_text(chunk) for chunk in chunks]
    return {
        "format": "naive",
        "background_rows": 0,
        "entity_rows": 0,
        "unique_entities": 0,
        "relation_rows": 0,
        "unique_relation_pairs": 0,
        "duplicate_relation_pairs": 0,
        "unique_relation_entries": 0,
        "duplicate_relation_entries": 0,
        "source_rows": len(chunks),
        "unique_source_rows": len(set(fingerprints)),
        "duplicate_source_rows": len(chunks) - len(set(fingerprints)),
    }


def summarize_context(context: str, model_name: str) -> dict:
    if context is None:
        return {
            "format": "missing",
            "background_rows": 0,
            "entity_rows": 0,
            "unique_entities": 0,
            "relation_rows": 0,
            "unique_relation_pairs": 0,
            "duplicate_relation_pairs": 0,
            "unique_relation_entries": 0,
            "duplicate_relation_entries": 0,
            "source_rows": 0,
            "unique_source_rows": 0,
            "duplicate_source_rows": 0,
            "context_chars": 0,
            "context_tokens": 0,
            "missing_context": 1,
        }
    summary = (
        parse_hierarchical_context(context)
        if any(header in context for header in SECTION_HEADERS["entities"])
        else parse_naive_context(context)
    )
    summary["context_chars"] = len(context)
    summary["context_tokens"] = len(encode_string_by_tiktoken(context, model_name=model_name))
    summary["missing_context"] = 0
    return summary


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


def aggregate_metric(query_rows: list[dict], key: str) -> dict:
    off_values = [row["off"][key] for row in query_rows]
    on_values = [row["on"][key] for row in query_rows]
    deltas = [row["delta"][key] for row in query_rows]
    return {
        "off_mean": sum(off_values) / len(off_values) if off_values else 0.0,
        "on_mean": sum(on_values) / len(on_values) if on_values else 0.0,
        "delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
        "queries_reduced": sum(1 for value in deltas if value < 0),
        "queries_unchanged": sum(1 for value in deltas if value == 0),
        "queries_increased": sum(1 for value in deltas if value > 0),
    }


def aggregate_summary(query_rows: list[dict]) -> dict:
    metrics = [
        "context_tokens",
        "context_chars",
        "background_rows",
        "entity_rows",
        "relation_rows",
        "source_rows",
        "duplicate_source_rows",
        "duplicate_relation_entries",
        "missing_context",
    ]
    summary = {
        "query_count": len(query_rows),
        "metrics": {metric: aggregate_metric(query_rows, metric) for metric in metrics},
    }
    if query_rows:
        summary["mean_relative_token_change"] = sum(
            row["relative_change"]["context_tokens"] for row in query_rows
        ) / len(query_rows)
        summary["mean_relative_source_change"] = sum(
            row["relative_change"]["source_rows"] for row in query_rows
        ) / len(query_rows)
    else:
        summary["mean_relative_token_change"] = 0.0
        summary["mean_relative_source_change"] = 0.0
    return summary


def summarize_judge_results(result_path: Path, query_count: int) -> dict:
    eval_path = result_path.with_name(result_path.stem + "_result_openai.jsonl")
    evaluations = []
    with eval_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                evaluations.append(json.loads(line))

    criteria = [
        "Comprehensiveness",
        "Empowerment",
        "Diversity",
        "Overall Winner",
    ]
    summary = {
        "judge_expected_count": query_count * 2,
        "judge_record_count": len(evaluations),
        "judge_missing_count": query_count * 2 - len(evaluations),
    }
    for criterion in criteria:
        off_wins = 0
        on_wins = 0
        for index, item in enumerate(evaluations):
            winner = item[criterion]["Winner"]
            off_is_answer1 = index < query_count
            if winner == "Answer 1":
                if off_is_answer1:
                    off_wins += 1
                else:
                    on_wins += 1
            elif winner == "Answer 2":
                if off_is_answer1:
                    on_wins += 1
                else:
                    off_wins += 1
        total = len(evaluations)
        summary[criterion] = {
            "off": safe_ratio(off_wins, total),
            "on": safe_ratio(on_wins, total),
            "count": total,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="mix")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="hi",
        help="hi / naive / hi_global / hi_local / hi_bridge / hi_nobridge",
    )
    parser.add_argument("--working-dir", type=str, default="")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--skip-answers", action="store_true")
    parser.add_argument("--judge-openai", action="store_true")
    args = parser.parse_args()

    dataset_root = dataset_dir(args.dataset)
    input_path = dataset_root / f"{args.dataset}.jsonl"
    working_dir = args.working_dir or str(dataset_root / "work_dir")
    output_prefix = args.output_prefix or str(
        dataset_root / f"{args.dataset}_{args.mode}_lightweight_dedup_q{args.max_queries}"
    )

    graph_func = HiRAG(
        working_dir=working_dir,
        enable_hierachical_mode=args.mode != "naive",
        embedding_func_max_async=4,
        enable_naive_rag=args.mode == "naive",
    )
    model_name = config_value(config, "openai", "model", "tiktoken_model_name", default="gpt-4o")

    queries = load_queries(input_path, args.max_queries)
    query_rows = []
    answer_rows_off = []
    answer_rows_on = []

    for index, query in enumerate(tqdm(queries, desc="A/B dedup eval"), start=1):
        off_context = graph_func.query(
            query=query,
            param=QueryParam(mode=args.mode, enable_dedup=False, only_need_context=True),
        )
        on_context = graph_func.query(
            query=query,
            param=QueryParam(mode=args.mode, enable_dedup=True, only_need_context=True),
        )
        off_metrics = summarize_context(off_context, model_name=model_name)
        on_metrics = summarize_context(on_context, model_name=model_name)
        delta_metrics = {
            key: on_metrics[key] - off_metrics[key]
            for key in off_metrics.keys()
            if key not in {"format"}
        }
        relative_change = {
            key: safe_ratio(on_metrics[key] - off_metrics[key], off_metrics[key])
            for key in (
                "context_tokens",
                "context_chars",
                "background_rows",
                "entity_rows",
                "relation_rows",
                "source_rows",
            )
        }

        row = {
            "index": index,
            "query": query,
            "off": off_metrics,
            "on": on_metrics,
            "delta": delta_metrics,
            "relative_change": relative_change,
        }
        query_rows.append(row)

        if not args.skip_answers:
            off_answer = graph_func.query(
                query=query,
                param=QueryParam(mode=args.mode, enable_dedup=False),
            )
            on_answer = graph_func.query(
                query=query,
                param=QueryParam(mode=args.mode, enable_dedup=True),
            )
            answer_rows_off.append({"query": query, "answer": off_answer})
            answer_rows_on.append({"query": query, "answer": on_answer})

    metrics_path = Path(output_prefix + "_metrics.jsonl")
    summary_path = Path(output_prefix + "_summary.json")
    query_path = Path(output_prefix + "_queries.jsonl")
    write_jsonl(metrics_path, query_rows)
    write_jsonl(query_path, [{"input": row["query"]} for row in query_rows])

    summary = aggregate_summary(query_rows)
    outputs = {
        "metrics": str(metrics_path),
        "queries": str(query_path),
    }

    if not args.skip_answers:
        off_answers_path = Path(output_prefix + "_answers_off.jsonl")
        on_answers_path = Path(output_prefix + "_answers_on.jsonl")
        write_jsonl(off_answers_path, answer_rows_off)
        write_jsonl(on_answers_path, answer_rows_on)
        outputs["answers_off"] = str(off_answers_path)
        outputs["answers_on"] = str(on_answers_path)

        if args.judge_openai:
            judge_path = Path(output_prefix + "_judge.jsonl")
            eval_oq_openai(
                query_file=str(query_path),
                result1_file=str(off_answers_path),
                result2_file=str(on_answers_path),
                output_file_path=str(judge_path),
            )
            outputs["judge"] = str(judge_path.with_name(judge_path.stem + "_result_openai.jsonl"))
            summary["judge_openai"] = summarize_judge_results(judge_path, query_count=len(query_rows))

    summary["outputs"] = outputs
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
