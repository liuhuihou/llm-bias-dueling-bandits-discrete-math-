from __future__ import annotations

import csv
import json
from pathlib import Path


def build_markdown_summary(stat_report_path: Path, output_path: Path) -> None:
    with stat_report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    lines = ["# Statistical Summary", "", "| Scenario | Baseline | p-value | Mean Improvement |", "|---|---:|---:|---:|"]

    for scenario, detail in report.items():
        sig = detail.get("significance_vs_dbs", {})
        for baseline, v in sig.items():
            p_value = float(v.get("p_value", 1.0))
            improve = float(v.get("mean_improvement", 0.0))
            lines.append(f"| {scenario} | {baseline} | {p_value:.4f} | {improve:.4f} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_latex_table_from_robustness(robustness_csv: Path, output_tex: Path) -> None:
    rows = []
    with robustness_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Relative final regret across bias scenarios (lower is better).}",
        "\\begin{tabular}{l l c}",
        "\\toprule",
        "Scenario & Algorithm & Relative Final Regret \\\\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(
            f"{row['scenario']} & {row['algorithm']} & {float(row['relative_final_regret']):.3f} \\\\"  # noqa: E501
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines), encoding="utf-8")
