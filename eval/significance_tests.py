"""Statistical significance tests for all LLM-as-judge experiments.

Computes for each experiment and criterion:
  - Win counts (improved vs baseline)
  - Win rate
  - Binomial test (two-sided, H0: p=0.5)
  - Wilson score 95% confidence interval
  - Effect size (Cohen's h)

Usage:
    cd eval/
    python significance_tests.py
    python significance_tests.py --format markdown > ../paper_experiments/08_SIGNIFICANCE_TESTS.md
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Statistics (no scipy dependency)
# ---------------------------------------------------------------------------

def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Exact binomial CDF via summation. Fine for n <= 300."""
    from math import comb
    total = 0.0
    for i in range(k + 1):
        total += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def binomial_test_two_sided(successes: int, trials: int, p0: float = 0.5) -> float:
    """Two-sided binomial test. Returns p-value."""
    if trials == 0:
        return 1.0
    # P(X <= successes) and P(X >= successes)
    p_left = _binomial_cdf(successes, trials, p0)
    p_right = 1.0 - _binomial_cdf(successes - 1, trials, p0)
    return min(1.0, 2.0 * min(p_left, p_right))


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if trials == 0:
        return (0.0, 1.0)
    p_hat = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p_hat + z ** 2 / (2 * trials)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * trials)) / trials) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


# ---------------------------------------------------------------------------
# Judge result parsing
# ---------------------------------------------------------------------------

CRITERIA = ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]


@dataclass
class ExperimentResult:
    name: str
    label_improved: str
    label_baseline: str
    query_count: int
    judge_records: int
    criteria: dict  # criterion -> {improved_wins, baseline_wins, total}


def parse_judge_file(
    path: Path,
    query_count: int,
    label_baseline: str = "baseline",
    label_improved: str = "improved",
) -> ExperimentResult:
    """Parse a judge result JSONL. First query_count records have baseline as Answer 1,
    next query_count have it as Answer 2."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    criteria_counts = {}
    for criterion in CRITERIA:
        improved_wins = 0
        baseline_wins = 0
        for idx, item in enumerate(records):
            winner = item[criterion]["Winner"]
            baseline_is_answer1 = idx < query_count
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
            # ties are ignored (neither wins)
        criteria_counts[criterion] = {
            "improved_wins": improved_wins,
            "baseline_wins": baseline_wins,
            "ties": len(records) - improved_wins - baseline_wins,
            "total": improved_wins + baseline_wins,  # excluding ties
        }

    return ExperimentResult(
        name=path.stem,
        label_improved=label_improved,
        label_baseline=label_baseline,
        query_count=query_count,
        judge_records=len(records),
        criteria=criteria_counts,
    )


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

def get_experiments(data_dir: Path) -> list[ExperimentResult]:
    experiments = []
    qc = 130  # all experiments use 130 queries

    defs = [
        ("mix_hi_hybrid_retrieval_q130_judge_result_openai.jsonl",
         "Hybrid RRF (k=20) vs Vector-only", "hybrid_rrf", "vector_only"),

        ("mix_hi_ablation_q130_judge_vector_only_vs_vector_bm25_result_openai.jsonl",
         "Vector+BM25 vs Vector-only", "vector_bm25", "vector_only"),

        ("mix_hi_ablation_q130_judge_vector_only_vs_vector_pr_result_openai.jsonl",
         "Vector+PageRank vs Vector-only", "vector_pr", "vector_only"),

        ("mix_hi_ablation_q130_judge_vector_only_vs_vector_bm25_pr_result_openai.jsonl",
         "Vector+BM25+PR vs Vector-only", "vector_bm25_pr", "vector_only"),

        ("mix_hi_tokennorm_bk20_hk10_q130_judge_result_openai.jsonl",
         "Hybrid RRF (k=10) vs Vector-only (k=20)", "hybrid_k10", "vector_k20"),

        ("mix_hi_tokennorm_bk20_hk7_q130_judge_result_openai.jsonl",
         "Hybrid RRF (k=7) vs Vector-only (k=20)", "hybrid_k7", "vector_k20"),

        ("mix_hi_mmr_lam07_q130_judge_result_openai.jsonl",
         "MMR λ=0.7 vs Hybrid baseline", "mmr_07", "hybrid_baseline"),

        ("mix_hi_mmr_lam09_q130_judge_result_openai.jsonl",
         "MMR λ=0.9 vs Hybrid baseline", "mmr_09", "hybrid_baseline"),

        ("mix_hi_ctxquality_k10_q130_judge_result_openai.jsonl",
         "Context Quality (prompt+rerank+filter) vs Hybrid RRF (k=10)", "ctx_quality", "hybrid_k10"),
    ]

    for filename, name, label_imp, label_base in defs:
        fpath = data_dir / filename
        if fpath.exists():
            exp = parse_judge_file(fpath, qc, label_base, label_imp)
            exp.name = name
            experiments.append(exp)
        else:
            print(f"WARNING: {fpath} not found, skipping", file=sys.stderr)

    return experiments


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(experiments: list[ExperimentResult], fmt: str = "text"):
    if fmt == "markdown":
        print("# Statistical Significance Tests")
        print()
        print("All tests computed from LLM-as-judge (GPT-4o) evaluations on 130 queries")
        print("from UltraDomain Mix. Each query judged twice with answer order swapped")
        print("(260 judge records per experiment). Ties excluded from win counts.")
        print()
        print("## Method")
        print()
        print("- **Binomial test** (two-sided): H₀: P(improved wins) = 0.5")
        print("- **Wilson score interval**: 95% confidence interval for win rate")
        print("- **Cohen's h**: effect size for two proportions (|h| > 0.2 = small, > 0.5 = medium, > 0.8 = large)")
        print("- Significance threshold: α = 0.05 (marked with *), α = 0.01 (marked with **), α = 0.001 (marked with ***)")
        print()

    for exp in experiments:
        if fmt == "markdown":
            print(f"## {exp.name}")
            print()
            print(f"Comparison: **{exp.label_improved}** vs **{exp.label_baseline}**")
            print(f"Judge records: {exp.judge_records} (from {exp.query_count} queries × 2 orderings)")
            print()
            print("| Criterion | Improved wins | Baseline wins | Ties | Win rate | 95% CI | p-value | Sig. | Cohen's h |")
            print("|---|---|---|---|---|---|---|---|---|")
        else:
            print(f"\n{'='*80}")
            print(f"  {exp.name}")
            print(f"  {exp.label_improved} vs {exp.label_baseline}")
            print(f"  ({exp.judge_records} judge records from {exp.query_count} queries)")
            print(f"{'='*80}")

        for criterion in CRITERIA:
            c = exp.criteria[criterion]
            iw = c["improved_wins"]
            bw = c["baseline_wins"]
            ties = c["ties"]
            total = c["total"]

            if total == 0:
                continue

            win_rate = iw / total
            ci_lo, ci_hi = wilson_ci(iw, total)
            p_val = binomial_test_two_sided(iw, total)
            h = cohens_h(win_rate, 0.5)

            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "ns"

            if fmt == "markdown":
                print(f"| {criterion} | {iw} | {bw} | {ties} | {win_rate:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {p_val:.4f} | {sig} | {h:+.3f} |")
            else:
                print(f"  {criterion:25s}  wins={iw:3d}/{bw:3d}  ties={ties:2d}  "
                      f"rate={win_rate:.3f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
                      f"p={p_val:.4f} {sig:3s}  h={h:+.3f}")

        if fmt == "markdown":
            print()

    # --- Summary table ---
    if fmt == "markdown":
        print("## Summary: Overall Winner Significance")
        print()
        print("| Experiment | Win rate | 95% CI | p-value | Significant? | Effect size |")
        print("|---|---|---|---|---|---|")

    for exp in experiments:
        c = exp.criteria["Overall Winner"]
        total = c["total"]
        if total == 0:
            continue
        iw = c["improved_wins"]
        win_rate = iw / total
        ci_lo, ci_hi = wilson_ci(iw, total)
        p_val = binomial_test_two_sided(iw, total)
        h = cohens_h(win_rate, 0.5)

        if p_val < 0.001:
            sig = "Yes (p<0.001)"
        elif p_val < 0.01:
            sig = "Yes (p<0.01)"
        elif p_val < 0.05:
            sig = "Yes (p<0.05)"
        else:
            sig = "No"

        if abs(h) < 0.2:
            eff = "negligible"
        elif abs(h) < 0.5:
            eff = "small"
        elif abs(h) < 0.8:
            eff = "medium"
        else:
            eff = "large"

        if fmt == "markdown":
            print(f"| {exp.name} | {win_rate:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {p_val:.4f} | {sig} | {h:+.3f} ({eff}) |")

    if fmt == "markdown":
        print()
        print("## Interpretation Guide")
        print()
        print("- **Win rate > 0.5**: improved variant wins more often")
        print("- **95% CI not containing 0.5**: result is significant at α=0.05")
        print("- **Cohen's h**: positive = improved wins more; |h| > 0.2 = practically meaningful")
        print("- **p-value**: probability of observing this result if both variants are equally good")
        print()
        print("## Script")
        print()
        print("```bash")
        print("cd eval/")
        print("python significance_tests.py                    # text output")
        print("python significance_tests.py --format markdown  # markdown tables")
        print("```")


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    parser.add_argument("--data-dir", default="")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent / "datasets" / "mix"

    experiments = get_experiments(data_dir)
    if not experiments:
        print("No experiment data found!", file=sys.stderr)
        sys.exit(1)

    print_results(experiments, fmt=args.format)


if __name__ == "__main__":
    main()
