import json
import argparse
from tqdm import tqdm
from hirag import HiRAG, QueryParam
import os

from _common import config_value, dataset_dir, load_config


config = load_config()
os.environ["OPENAI_API_KEY"] = config_value(config, "openai", "api_key")
openai_base_url = config_value(config, "openai", "base_url", default="")
if openai_base_url and openai_base_url != "***":
    os.environ["OPENAI_BASE_URL"] = openai_base_url

MAX_QUERIES = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="mix")
    parser.add_argument("-m", "--mode", type=str, default="hi", help="hi / naive / hi_global / hi_local / hi_bridge / hi_nobridge")
    args = parser.parse_args()
    
    DATASET = args.dataset
    dataset_root = dataset_dir(DATASET)
    input_path = dataset_root / f"{DATASET}.jsonl"
    output_path = dataset_root / f"{DATASET}_{args.mode}_result.jsonl"
    graph_func = HiRAG(
        working_dir=str(dataset_root / "work_dir"),
        enable_hierachical_mode=args.mode != "naive",
        embedding_func_max_async=4,
        enable_naive_rag=args.mode == "naive",
    )

    query_list = []
    with open(input_path, encoding="utf-8", mode="r") as f:      # get context
        lines = f.readlines()
        for item in lines:
            item_dict = json.loads(item)
            query_list.append(item_dict["input"])
    query_list = query_list[:MAX_QUERIES]
    answer_list = []

    print(f"Perform {args.mode} search:")
    for query in tqdm(query_list):
        tqdm.write(f"Q: {query}")
        answer = graph_func.query(query=query, param=QueryParam(mode=args.mode))
        tqdm.write(f"A: {answer} \n ################################################################################################")
        answer_list.append(answer)
    
    result_to_write = []
    for query, answer in zip(query_list, answer_list):
        result_to_write.append({"query": query, "answer": answer})
    with open(output_path, "w") as f:
        for item in result_to_write:
            f.write(json.dumps(item) + "\n")
        
