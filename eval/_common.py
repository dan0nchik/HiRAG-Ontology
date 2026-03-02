from pathlib import Path
import sys

import yaml


EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
DATASETS_DIR = EVAL_DIR / "datasets"
CONFIG_PATH = REPO_ROOT / "config.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def config_value(config: dict, section: str, *keys: str, default=None):
    section_config = config.get(section, {})
    for key in keys:
        if key in section_config:
            return section_config[key]
    if default is not None:
        return default
    tried = ", ".join(keys)
    raise KeyError(f"Missing config value for {section}: tried {tried}")


def dataset_dir(dataset: str) -> Path:
    return DATASETS_DIR / dataset


def usable_base_url(url: str | None) -> str | None:
    if not url:
        return None
    if url.startswith(("http://", "https://")):
        return url
    return None
