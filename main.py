from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from hirag import HiRAG

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_WORKDIR_ROOT = PROJECT_ROOT / "hirag_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from a PDF in ./data, save it as .txt, "
            "and insert the content into a HiRAG workspace."
        )
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Path to the PDF file. Defaults to the first *.pdf inside the data directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory that stores PDFs (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"HiRAG configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the generated .txt file. Defaults to <pdf_path>.txt.",
    )
    parser.add_argument(
        "--question",
        action="append",
        help=(
            "Question to ask using HiRAG after insertion. "
            "Repeat the flag to ask multiple questions."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["hi", "hi_bridge", "hi_local", "hi_global", "hi_nobridge", "naive"],
        default="hi",
        help="Retrieval mode to use for QA (default: hi).",
    )
    return parser.parse_args()


def resolve_pdf_path(pdf_arg: Path | None, data_dir: Path) -> Path:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if pdf_arg:
        pdf_path = pdf_arg.expanduser().resolve()
    else:
        pdf_candidates = sorted(data_dir.glob("*.pdf"))
        if not pdf_candidates:
            raise FileNotFoundError(f"No PDF files found inside {data_dir}")
        pdf_path = pdf_candidates[0]

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    return pdf_path


def extract_pdf_text(pdf_path: Path) -> str:
    """Return the concatenated text of every page in the PDF."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pypdf is required to extract text. Install it via `pip install pypdf`."
        ) from exc

    reader = PdfReader(str(pdf_path))
    page_text: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        page_text.append(f"### Page {page_number}\n{text}")

    combined_text = "\n\n".join(page_text).strip()
    if not combined_text:
        raise ValueError(f"No extractable text found in {pdf_path.name}")
    return combined_text


def write_text_file(text: str, output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def load_hirag_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_hirag_instance(config: dict[str, Any], fallback_workdir: Path) -> "HiRAG":
    from hirag import HiRAG  # imported lazily to avoid heavy deps on --help etc.
    hirag_config = config.get("hirag", {})
    declared_workdir = str(hirag_config.get("working_dir", "")).strip()
    if not declared_workdir or declared_workdir.lower() == "your_work_dir":
        working_dir = fallback_workdir
    else:
        working_dir = Path(declared_workdir).expanduser()

    working_dir.parent.mkdir(parents=True, exist_ok=True)

    hirag_kwargs: dict[str, Any] = {"working_dir": str(working_dir)}
    for key in (
        "enable_llm_cache",
        "enable_hierachical_mode",
        "embedding_batch_num",
        "embedding_func_max_async",
        "enable_naive_rag",
    ):
        if key in hirag_config:
            hirag_kwargs[key] = hirag_config[key]

    return HiRAG(**hirag_kwargs)


def run_question_answering(graph_func: "HiRAG", question: str, mode: str) -> Any:
    """Execute a HiRAG query using the requested mode."""
    from hirag import QueryParam  # local import to avoid heavy deps during --help

    param = QueryParam(mode=mode)
    return graph_func.query(question, param=param)


def main() -> int:
    args = parse_args()
    pdf_path = resolve_pdf_path(args.pdf, args.data_dir)
    extracted_text = extract_pdf_text(pdf_path)

    output_path = args.output or pdf_path.with_suffix(".txt")
    output_txt = write_text_file(extracted_text, Path(output_path))

    config = load_hirag_config(args.config)
    fallback_workdir = (DEFAULT_WORKDIR_ROOT / pdf_path.stem).resolve()
    graph_func = build_hirag_instance(config, fallback_workdir)

    graph_func.insert(extracted_text)

    print(f"Saved extracted text to: {output_txt}")
    print(f"Inserted '{pdf_path.name}' into HiRAG workspace: {graph_func.working_dir}")

    if args.question:
        for idx, question in enumerate(args.question, start=1):
            question = question.strip()
            if not question:
                continue
            print(f"\n[QA #{idx}] Question: {question}")
            try:
                answer = run_question_answering(graph_func, question, args.mode)
            except ValueError as exc:
                raise SystemExit(f"Failed to run QA: {exc}") from exc
            print(f"[QA #{idx}] Answer:\n{answer}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
