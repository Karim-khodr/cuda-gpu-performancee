#!/usr/bin/env python3
"""Generate the normalized headline-performance SVG from authoritative data."""

from __future__ import annotations

import csv
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "results" / "headline_performance.csv"
OUTPUT_PATH = ROOT / "results" / "headline_performance.svg"
FIELDS = [
    "workload",
    "representative_size",
    "run",
    "baseline_label",
    "baseline_ms",
    "optimized_label",
    "optimized_ms",
]
WORKLOAD_ORDER = ["Reduction", "Transpose", "Convolution"]


def load_rows() -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"unexpected CSV columns: {reader.fieldnames}")
        for raw in reader:
            baseline = Decimal(raw["baseline_ms"])
            optimized = Decimal(raw["optimized_ms"])
            if baseline <= 0 or optimized <= 0:
                raise ValueError("kernel times must be positive")
            grouped[raw["workload"]].append(
                {
                    "size": raw["representative_size"],
                    "run": int(raw["run"]),
                    "baseline_label": raw["baseline_label"],
                    "optimized_label": raw["optimized_label"],
                    "ratio": optimized / baseline,
                }
            )

    if set(grouped) != set(WORKLOAD_ORDER):
        raise ValueError(f"unexpected workloads: {sorted(grouped)}")
    for workload in WORKLOAD_ORDER:
        grouped[workload].sort(key=lambda row: int(row["run"]))
        if [row["run"] for row in grouped[workload]] != [1, 2]:
            raise ValueError(f"{workload} must contain authoritative runs 1 and 2")
        identity = {
            (
                row["size"],
                row["baseline_label"],
                row["optimized_label"],
            )
            for row in grouped[workload]
        }
        if len(identity) != 1:
            raise ValueError(f"inconsistent labels or size for {workload}")
    return grouped


def fmt(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.001')):.3f}"


def generate_svg(grouped: dict[str, list[dict[str, object]]]) -> str:
    width, height = 960, 560
    chart_left, chart_right = 300, 900
    chart_top, chart_bottom = 160, 450
    row_y = {"Reduction": 215, "Transpose": 315, "Convolution": 415}
    all_ratios = [
        row["ratio"] for workload in WORKLOAD_ORDER for row in grouped[workload]
    ]
    tick = Decimal("0.25")
    axis_max = (max(max(all_ratios), Decimal("1")) / tick).to_integral_value(
        rounding=ROUND_CEILING
    ) * tick

    def x_position(value: Decimal) -> float:
        return chart_left + float(value / axis_max) * (chart_right - chart_left)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '  <title id="title">Relative kernel time across three CUDA experiments</title>',
        '  <desc id="description">Optimized mean kernel time divided by baseline mean kernel time for two authoritative runs of reduction, transpose, and convolution. Lower values are better; convolution is above baseline.</desc>',
        "  <style>",
        "    :root { --bg: #ffffff; --fg: #172033; --muted: #5b6475; --grid: #d8dde7; --improvement: #16775b; --regression: #b85c16; --baseline: #526176; }",
        "    @media (prefers-color-scheme: dark) { :root { --bg: #111827; --fg: #f3f4f6; --muted: #c2c9d6; --grid: #3b4558; --improvement: #55d6a8; --regression: #ffad66; --baseline: #aeb8c8; } }",
        "    text { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: var(--fg); }",
        "    .title { font-size: 24px; font-weight: 600; }",
        "    .subtitle, .axis, .note { fill: var(--muted); font-size: 13px; }",
        "    .workload { font-size: 17px; font-weight: 600; }",
        "    .detail, .value { font-size: 13px; }",
        "    .value { font-weight: 600; }",
        "    .grid { stroke: var(--grid); stroke-width: 1; }",
        "    .baseline { stroke: var(--baseline); stroke-width: 2; stroke-dasharray: 5 5; }",
        "    .range-improvement { stroke: var(--improvement); fill: var(--improvement); }",
        "    .range-regression { stroke: var(--regression); fill: var(--regression); }",
        "  </style>",
        '  <rect width="960" height="560" fill="var(--bg)"/>',
        '  <text x="30" y="42" class="title">Relative kernel time across three CUDA experiments</text>',
        '  <text x="30" y="68" class="subtitle">Each baseline = 1.0 · lower is better · points are the two authoritative runs</text>',
    ]

    current = Decimal("0")
    while current <= axis_max:
        x = x_position(current)
        css_class = "baseline" if current == Decimal("1") else "grid"
        lines.append(
            f'  <line x1="{x:.2f}" y1="{chart_top}" x2="{x:.2f}" y2="{chart_bottom}" class="{css_class}"/>'
        )
        lines.append(
            f'  <text x="{x:.2f}" y="{chart_bottom + 25}" text-anchor="middle" class="axis">{fmt(current)}</text>'
        )
        current += tick

    baseline_x = x_position(Decimal("1"))
    lines.append(
        f'  <text x="{baseline_x:.2f}" y="{chart_top - 12}" text-anchor="middle" class="axis">baseline 1.0</text>'
    )

    for workload in WORKLOAD_ORDER:
        rows = grouped[workload]
        ratios = [row["ratio"] for row in rows]
        low, high = min(ratios), max(ratios)
        mean = sum(ratios, Decimal("0")) / Decimal(len(ratios))
        y = row_y[workload]
        size = escape(str(rows[0]["size"]))
        optimized_label = escape(str(rows[0]["optimized_label"]))
        baseline_label = escape(str(rows[0]["baseline_label"]))
        style = "range-improvement" if mean < 1 else "range-regression"
        outcome = "lower relative time" if mean < 1 else "higher relative time"
        label_x = min(x_position(high) + 14, width - 105)
        lines.extend(
            [
                f'  <text x="30" y="{y - 8}" class="workload">{escape(workload)}</text>',
                f'  <text x="30" y="{y + 14}" class="detail">{size} · {optimized_label} / {baseline_label}</text>',
                f'  <line x1="{x_position(low):.2f}" y1="{y}" x2="{x_position(high):.2f}" y2="{y}" class="{style}" stroke-width="5" stroke-linecap="round"/>',
                f'  <circle cx="{x_position(ratios[0]):.2f}" cy="{y - 7}" r="7" class="{style}"/>',
                f'  <rect x="{x_position(ratios[1]) - 6:.2f}" y="{y + 1}" width="12" height="12" class="{style}" transform="rotate(45 {x_position(ratios[1]):.2f} {y + 7})"/>',
                f'  <text x="{label_x:.2f}" y="{y - 11}" class="value">{fmt(low)}–{fmt(high)}×</text>',
                f'  <text x="{label_x:.2f}" y="{y + 10}" class="detail">{outcome}</text>',
            ]
        )

    lines.extend(
        [
            f'  <text x="{(chart_left + chart_right) / 2:.2f}" y="{chart_bottom + 55}" text-anchor="middle" class="axis">optimized mean kernel time / baseline mean kernel time</text>',
            '  <circle cx="327" cy="522" r="6" fill="var(--baseline)"/>',
            '  <text x="340" y="527" class="note">Run 1</text>',
            '  <rect x="404" y="516" width="11" height="11" fill="var(--baseline)" transform="rotate(45 409.5 521.5)"/>',
            '  <text x="424" y="527" class="note">Run 2</text>',
            '  <text x="510" y="527" class="note">Normalized independently; not an absolute runtime comparison.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    grouped = load_rows()
    svg = generate_svg(grouped)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(svg)
    for workload in WORKLOAD_ORDER:
        ratios = ", ".join(fmt(row["ratio"]) for row in grouped[workload])
        print(f"{workload}: {ratios}")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
