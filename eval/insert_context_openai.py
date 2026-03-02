import os
import json
import time
from hirag import HiRAG, QueryParam
from _common import config_value, dataset_dir, load_config


config = load_config()
os.environ["OPENAI_API_KEY"] = config_value(config, "openai", "api_key")
openai_base_url = config_value(config, "openai", "base_url", default="")
if openai_base_url and openai_base_url != "***":
    os.environ["OPENAI_BASE_URL"] = openai_base_url

DATASET = "mix"
dataset_root = dataset_dir(DATASET)
file_path = dataset_root / f"{DATASET}_unique_contexts.json"

graph_func = HiRAG(
    working_dir=str(dataset_root / "work_dir_hi"),
    enable_hierachical_mode=True, 
    embedding_func_max_async=4,
    enable_naive_rag=True)

with open(file_path, mode="r") as f:
    unique_contexts = json.load(f)
    graph_func.insert(unique_contexts)
