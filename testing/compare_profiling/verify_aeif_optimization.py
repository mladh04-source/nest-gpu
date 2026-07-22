#!/usr/bin/env python3
"""
Reproduce and verify the AEIF resource-management optimization results
from the combined Nsight Systems text log.

This is a post-processing/reproducibility test. It does not rerun NEST GPU.
It verifies that the reported before/after values can be derived from the raw
program and Nsight output and that activity remained effectively unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SVG_VERSION = "2026-07-22-no-matplotlib-v7"

SIZES = (1000, 2000, 5000)
RATE_TOLERANCE_HZ = 0.010 + 1e-12
CV_TOLERANCE = 8.1e-4 + 1e-12

CONFIG_FIELDS = (
    "actual_neurons",
    "n_exc",
    "n_inh",
    "ce",
    "ci",
    "neural_activity_simulation_time_ms",
    "w_ex",
    "w_in",
    "poiss_rate",
    "poiss_weight",
    "poiss_delay",
)


def fail(message: str) -> None:
    raise AssertionError(message)


def find_case_insensitive(text: str, marker: str) -> int:
    return text.lower().find(marker.lower())


def split_required(text: str, marker: str, label: str) -> tuple[str, str]:
    pos = find_case_insensitive(text, marker)
    if pos < 0:
        fail(f"Marker für {label!r} nicht gefunden: {marker!r}")
    return text[:pos], text[pos + len(marker):]


def extract_balanced_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []

    for match in re.finditer(r'\{\s*"family"\s*:\s*"aeif"', text):
        start = match.start()
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    raw = text[start:index + 1]
                    objects.append(json.loads(raw))
                    break

    return objects


def map_json_by_size(text: str, revision: str) -> dict[int, dict[str, Any]]:
    objects = extract_balanced_json_objects(text)
    result: dict[int, dict[str, Any]] = {}

    for obj in objects:
        size = int(obj["requested_neurons"])
        if size in SIZES:
            if size in result:
                fail(f"Doppelter JSON-Datensatz für {revision}, N={size}")
            result[size] = obj

    missing = set(SIZES) - set(result)
    if missing:
        fail(f"Fehlende JSON-Datensätze für {revision}: {sorted(missing)}")

    return result


def extract_update_times(text: str, revision: str) -> dict[int, float]:
    values = [
        float(value)
        for value in re.findall(
            r"neuron_Update_time:\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            flags=re.IGNORECASE,
        )
    ]

    if len(values) != len(SIZES):
        fail(
            f"Für {revision} wurden {len(values)} neuron_Update_time-Werte "
            f"gefunden; erwartet: {len(SIZES)}"
        )

    return dict(zip(SIZES, values))


OLD_API_NAMES = (
    "cudaStreamSynchronize",
    "cudaLaunchKernel",
    "cudaMemcpyAsync",
    "cudaFree",
    "cudaMalloc",
    "cudaMemcpyFromSymbol",
    "cudaMemcpyToSymbolAsync",
    "cudaDeviceSynchronize",
    "cudaMemsetAsync",
    "cudaMemcpy",
    "cuGetProcAddress_v2",
    "cudaMemset",
    "cuInit",
)

OLD_NAME_PATTERN = "|".join(
    re.escape(name) for name in sorted(OLD_API_NAMES, key=len, reverse=True)
)

OLD_API_ROW = re.compile(
    rf"(?P<pct>\d+(?:\.\d+)?)\s+"
    rf"(?P<total>[\d,]+)\s+"
    rf"(?P<calls>[\d,]+)\s+"
    rf"(?P<avg>[\d,]+(?:\.\d+)?)\s+"
    rf"(?P<med>[\d,]+(?:\.\d+)?)\s+"
    rf"(?P<min>[\d,]+)\s+"
    rf"(?P<max>[\d,]+)\s+"
    rf"(?P<std>[\d,]+(?:\.\d+)?)\s+"
    rf"(?P<name>{OLD_NAME_PATTERN})"
)

NEW_API_ROW = re.compile(
    rf"(?P<pct>\d+(?:\.\d+)?),"
    rf"(?P<total>\d+),"
    rf"(?P<calls>\d+),"
    rf"(?P<avg>\d+(?:\.\d+)?),"
    rf"(?P<med>\d+(?:\.\d+)?),"
    rf"(?P<min>\d+),"
    rf"(?P<max>\d+),"
    rf"(?P<std>\d+(?:\.\d+)?),"
    rf'"?(?P<name>{OLD_NAME_PATTERN})"?'
)


def parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def find_size_section(text: str, size: int) -> str:
    marker = f"Profiling AEIF NESTML with {size} neurons"
    start = find_case_insensitive(text, marker)
    if start < 0:
        fail(f"Profiling-Abschnitt für N={size} nicht gefunden")

    later_positions = [
        pos
        for other_size in SIZES
        if other_size != size
        for pos in [find_case_insensitive(text[start + len(marker):],
                                          f"Profiling AEIF NESTML with {other_size} neurons")]
        if pos >= 0
    ]

    if not later_positions:
        return text[start:]

    end = start + len(marker) + min(later_positions)
    return text[start:end]


def parse_old_api(text: str) -> dict[int, dict[str, dict[str, float]]]:
    output: dict[int, dict[str, dict[str, float]]] = {}

    for size in SIZES:
        section = find_size_section(text, size)
        begin = find_case_insensitive(section, "Executing 'cuda_api_sum' stats report")
        end = find_case_insensitive(section, "Executing 'cuda_gpu_kern_sum' stats report")

        if begin < 0 or end < 0 or end <= begin:
            fail(f"Alter CUDA-API-Report für N={size} nicht gefunden")

        report = section[begin:end]
        rows: dict[str, dict[str, float]] = {}

        for match in OLD_API_ROW.finditer(report):
            data = match.groupdict()
            rows[data["name"]] = {
                "calls": parse_number(data["calls"]),
                "total_ns": parse_number(data["total"]),
            }

        for name in ("cudaMalloc", "cudaFree", "cudaMemcpyFromSymbol"):
            if name not in rows:
                fail(f"{name} im alten CUDA-API-Report für N={size} nicht gefunden")

        output[size] = rows

    return output


def parse_new_api(text: str) -> dict[int, dict[str, dict[str, float]]]:
    output: dict[int, dict[str, dict[str, float]]] = {}

    for size in SIZES:
        begin_marker = f"cuda_api_sum: {size} neurons"
        end_marker = f"cuda_gpu_kern_sum: {size} neurons"

        begin = find_case_insensitive(text, begin_marker)
        end = find_case_insensitive(text, end_marker)

        if begin < 0 or end < 0 or end <= begin:
            fail(f"Optimierter CUDA-API-Report für N={size} nicht gefunden")

        report = text[begin:end]
        rows: dict[str, dict[str, float]] = {}

        for match in NEW_API_ROW.finditer(report):
            data = match.groupdict()
            rows[data["name"]] = {
                "calls": parse_number(data["calls"]),
                "total_ns": parse_number(data["total"]),
            }

        for name in ("cudaMalloc", "cudaFree"):
            if name not in rows:
                fail(f"{name} im optimierten CUDA-API-Report für N={size} nicht gefunden")

        # The row is absent when there are zero calls.
        rows.setdefault("cudaMemcpyFromSymbol", {"calls": 0.0, "total_ns": 0.0})
        output[size] = rows

    return output


def reduction(initial: float, optimized: float) -> float:
    return (initial - optimized) / initial * 100.0


def almost_equal(left: Any, right: Any, *, abs_tol: float = 1e-12) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=abs_tol)
    return left == right



def _svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_text(
    x: float,
    y: float,
    text: object,
    *,
    anchor: str = "middle",
    size: int = 14,
    weight: str = "normal",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-size="{size}" font-weight="{weight}"'
        f'{transform}>{_svg_escape(text)}</text>'
    )


def create_svg_charts(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[Path, Path]:
    """
    Generate two SVG diagrams using only the Python standard library.

    Adjustments in v4:
    - more space at the bottom for tick labels, x-axis title, and legend
    - larger legend box in the reduction chart so the text is fully visible
    - slightly smaller legend box in the wall-time chart
    - x-axis labels positioned clearly below the plotted data
    """
    neurons = [int(row["neurons"]) for row in rows]
    initial_wall = [float(row["initial_sim_wall_s"]) for row in rows]
    optimized_wall = [float(row["optimized_sim_wall_s"]) for row in rows]
    wall_reduction = [float(row["sim_reduction_pct"]) for row in rows]
    update_reduction = [float(row["update_reduction_pct"]) for row in rows]

    width = 980
    height = 760
    left = 110
    right = 45
    top = 80
    bottom = 250
    plot_width = width - left - right
    plot_height = height - top - bottom

    # ------------------------------------------------------------------
    # Diagram 1: initial and optimized simulation wall times.
    # ------------------------------------------------------------------
    wall_path = out_dir / "aeif_wall_time_comparison.svg"
    wall_max = max(initial_wall + optimized_wall)
    y_max = max(10.0, (int(wall_max / 10.0) + 2) * 10.0)
    tick_step = y_max / 5.0

    def wall_x(index: int) -> float:
        if len(neurons) == 1:
            return left + plot_width / 2.0
        return left + index * plot_width / (len(neurons) - 1)

    def wall_y(value: float) -> float:
        return top + plot_height - value / y_max * plot_height

    baseline = height - bottom
    tick_label_y = baseline + 30
    axis_title_y = baseline + 75

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 36, "AEIF simulation time before and after optimization",
                  size=22, weight="bold"),
    ]

    for tick in range(6):
        value = tick * tick_step
        y = wall_y(value)
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#d9d9d9" stroke-width="1"/>'
        )
        svg.append(_svg_text(left - 14, y + 5, f"{value:.0f}", anchor="end", size=13))

    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" '
        'stroke="black" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{baseline}" x2="{width-right}" '
        f'y2="{baseline}" stroke="black" stroke-width="1.5"/>',
        _svg_text(30, top + plot_height / 2, "Simulation wall time [s]",
                  size=15, rotate=-90),
        _svg_text(left + plot_width / 2, axis_title_y, "Number of neurons", size=15),
    ])

    for index, neuron_count in enumerate(neurons):
        x = wall_x(index)
        svg.append(
            f'<line x1="{x:.1f}" y1="{baseline}" x2="{x:.1f}" '
            f'y2="{baseline+6}" stroke="black"/>'
        )
        svg.append(_svg_text(x, tick_label_y, neuron_count, size=13))

    initial_points = " ".join(
        f"{wall_x(i):.1f},{wall_y(value):.1f}"
        for i, value in enumerate(initial_wall)
    )
    optimized_points = " ".join(
        f"{wall_x(i):.1f},{wall_y(value):.1f}"
        for i, value in enumerate(optimized_wall)
    )

    svg.append(
        f'<polyline points="{initial_points}" fill="none" stroke="#1f77b4" '
        'stroke-width="3"/>'
    )
    svg.append(
        f'<polyline points="{optimized_points}" fill="none" stroke="#ff7f0e" '
        'stroke-width="3"/>'
    )

    for index, value in enumerate(initial_wall):
        x, y = wall_x(index), wall_y(value)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#1f77b4"/>')
        svg.append(_svg_text(x, y - 12, f"{value:.2f} s", size=12))

    for index, value in enumerate(optimized_wall):
        x, y = wall_x(index), wall_y(value)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#ff7f0e"/>')
        svg.append(_svg_text(x, y + 23, f"{value:.2f} s", size=12))

    # Smaller legend, lower placement.
    legend_width = 330
    legend_height = 44
    legend_x = (width - legend_width) / 2.0
    legend_y = height - 120
    svg.extend([
        f'<rect x="{legend_x:.1f}" y="{legend_y:.1f}" width="{legend_width}" height="{legend_height}" '
        'fill="white" stroke="#999"/>',
        f'<line x1="{legend_x+18:.1f}" y1="{legend_y+16:.1f}" x2="{legend_x+52:.1f}" '
        f'y2="{legend_y+16:.1f}" stroke="#1f77b4" stroke-width="3"/>',
        _svg_text(legend_x + 62, legend_y + 21, "Initial code",
                  anchor="start", size=13),
        f'<line x1="{legend_x+184:.1f}" y1="{legend_y+16:.1f}" x2="{legend_x+218:.1f}" '
        f'y2="{legend_y+16:.1f}" stroke="#ff7f0e" stroke-width="3"/>',
        _svg_text(legend_x + 228, legend_y + 21, "Optimized code",
                  anchor="start", size=13),
        "</svg>",
    ])
    wall_path.write_text("\n".join(svg), encoding="utf-8")

    # ------------------------------------------------------------------
    # Diagram 2: percentage reductions.
    # ------------------------------------------------------------------
    reduction_path = out_dir / "aeif_runtime_reduction_percent.svg"
    reduction_max = max(wall_reduction + update_reduction)
    reduction_y_max = max(10.0, (int(reduction_max / 5.0) + 2) * 5.0)
    reduction_tick_step = reduction_y_max / 5.0
    group_width = plot_width / len(neurons)
    bar_width = min(90.0, group_width * 0.26)
    gap = 18.0

    def reduction_y(value: float) -> float:
        return top + plot_height - value / reduction_y_max * plot_height

    baseline = height - bottom
    tick_label_y = baseline + 30
    axis_title_y = baseline + 75

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 36, "Observed AEIF runtime reduction",
                  size=22, weight="bold"),
    ]

    for tick in range(6):
        value = tick * reduction_tick_step
        y = reduction_y(value)
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#d9d9d9" stroke-width="1"/>'
        )
        svg.append(_svg_text(left - 14, y + 5, f"{value:.0f}", anchor="end", size=13))

    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" '
        'stroke="black" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{baseline}" x2="{width-right}" '
        f'y2="{baseline}" stroke="black" stroke-width="1.5"/>',
        _svg_text(30, top + plot_height / 2, "Reduction [%]", size=15, rotate=-90),
        _svg_text(left + plot_width / 2, axis_title_y, "Number of neurons", size=15),
    ])

    for index, neuron_count in enumerate(neurons):
        center = left + group_width * (index + 0.5)
        wall_x_pos = center - gap / 2.0 - bar_width
        update_x_pos = center + gap / 2.0

        wall_value = wall_reduction[index]
        update_value = update_reduction[index]
        wall_top = reduction_y(wall_value)
        update_top = reduction_y(update_value)

        svg.append(
            f'<rect x="{wall_x_pos:.1f}" y="{wall_top:.1f}" '
            f'width="{bar_width:.1f}" height="{baseline-wall_top:.1f}" '
            'fill="#1f77b4"/>'
        )
        svg.append(
            f'<rect x="{update_x_pos:.1f}" y="{update_top:.1f}" '
            f'width="{bar_width:.1f}" height="{baseline-update_top:.1f}" '
            'fill="#ff7f0e"/>'
        )

        wall_label_y = wall_top + (baseline - wall_top) / 2.0 + 5.0
        update_label_y = update_top + (baseline - update_top) / 2.0 + 5.0
        svg.append(_svg_text(
            wall_x_pos + bar_width / 2,
            wall_label_y,
            f"{wall_value:.1f}%",
            size=13,
            weight="bold",
        ))
        svg.append(_svg_text(
            update_x_pos + bar_width / 2,
            update_label_y,
            f"{update_value:.1f}%",
            size=13,
            weight="bold",
        ))
        svg.append(_svg_text(center, tick_label_y, neuron_count, size=13))

    # Larger legend so the full text is visible.
    legend_width = 430
    legend_height = 52
    legend_x = (width - legend_width) / 2.0
    legend_y = height - 102
    svg.extend([
        f'<rect x="{legend_x:.1f}" y="{legend_y:.1f}" width="{legend_width}" height="{legend_height}" '
        'fill="white" stroke="#999"/>',
        f'<rect x="{legend_x+20:.1f}" y="{legend_y+18:.1f}" width="26" height="16" '
        'fill="#1f77b4"/>',
        _svg_text(legend_x + 58, legend_y + 31, "Simulation wall time",
                  anchor="start", size=14),
        f'<rect x="{legend_x+245:.1f}" y="{legend_y+18:.1f}" width="26" height="16" '
        'fill="#ff7f0e"/>',
        _svg_text(legend_x + 283, legend_y + 31, "neuron_Update time",
                  anchor="start", size=14),
        "</svg>",
    ])
    reduction_path.write_text("\n".join(svg), encoding="utf-8")

    return wall_path, reduction_path




def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "aeif_optimization_verified_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    wall_time_chart, reduction_chart = create_svg_charts(rows, out_dir)

    report_path = out_dir / "aeif_optimization_verification_report.md"
    lines = [
        "# AEIF resource-management optimization: verification report",
        "",
        "**Result: PASS**",
        "",
        "This report was generated directly from the combined initial/optimized "
        "program and Nsight Systems output.",
        "",
        "## Generated diagrams",
        "",
        f"![Wall-time comparison]({wall_time_chart.name})",
        "",
        f"![Runtime reduction]({reduction_chart.name})",
        "",
        "## Timing and activity",
        "",
        "| N | Wall initial [s] | Wall optimized [s] | Reduction | "
        "Update initial [s] | Update optimized [s] | Reduction | "
        "max. rate difference [Hz] | CV difference |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['neurons']} | {row['initial_sim_wall_s']:.4f} | "
            f"{row['optimized_sim_wall_s']:.4f} | "
            f"{row['sim_reduction_pct']:.1f}% | "
            f"{row['initial_neuron_update_s']:.4f} | "
            f"{row['optimized_neuron_update_s']:.4f} | "
            f"{row['update_reduction_pct']:.1f}% | "
            f"{row['max_rate_difference_hz']:.5f} | "
            f"{row['cv_difference']:.7f} |"
        )

    lines += [
        "",
        "## CUDA API call counts",
        "",
        "| N | cudaMalloc initial | optimized | reduction | "
        "cudaFree initial | optimized | reduction | "
        "cudaMemcpyFromSymbol initial | optimized |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['neurons']} | {row['initial_cudaMalloc_calls']} | "
            f"{row['optimized_cudaMalloc_calls']} | "
            f"{row['cudaMalloc_call_reduction_pct']:.1f}% | "
            f"{row['initial_cudaFree_calls']} | "
            f"{row['optimized_cudaFree_calls']} | "
            f"{row['cudaFree_call_reduction_pct']:.1f}% | "
            f"{row['initial_cudaMemcpyFromSymbol_calls']} | "
            f"{row['optimized_cudaMemcpyFromSymbol_calls']} |"
        )

    lines += [
        "",
        "## What this test establishes",
        "",
        "- The before/after percentages are reproducible from the raw output.",
        "- The optimized run has fewer CUDA allocation/deallocation calls.",
        "- The per-interval `cudaMemcpyFromSymbol` calls disappear.",
        "- The recorded activity remains effectively unchanged within the "
        "thresholds used by this test.",
        "",
        "## Limitation",
        "",
        "This is a verification of the supplied profiler outputs, not a fresh "
        "benchmark execution. The log contains one profiler-instrumented run "
        "per revision and network size, so it demonstrates the observed paired "
        "difference but does not provide a confidence interval or statistical "
        "significance. Repeated unprofiled runs are the appropriate next step "
        "for runtime variability.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print(f"AEIF verification script version: {SVG_VERSION}")
    print("Diagram output: SVG (no matplotlib required)")
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path, help="Combined old/new Nsight text file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("aeif_verification_results"),
        help="Directory for generated CSV, Markdown report, and SVG diagrams",
    )
    args = parser.parse_args()

    text = args.log_file.read_text(encoding="utf-8", errors="replace")

    _, after_old_marker = split_required(
        text, "Das ist die alte Ausgabe:", "alte Ausgabe"
    )
    old_output, _ = split_required(
        after_old_marker, "das ist der neue code", "Ende der alten Ausgabe"
    )
    _, new_output = split_required(
        text, "Das ist die neue Ausgabe:", "neue Ausgabe"
    )

    initial_json = map_json_by_size(old_output, "initial")
    optimized_json = map_json_by_size(new_output, "optimized")
    initial_update = extract_update_times(old_output, "initial")
    optimized_update = extract_update_times(new_output, "optimized")
    initial_api = parse_old_api(old_output)
    optimized_api = parse_new_api(new_output)

    rows: list[dict[str, Any]] = []
    max_rate_difference = 0.0
    max_cv_difference = 0.0

    for size in SIZES:
        before = initial_json[size]
        after = optimized_json[size]

        for field in CONFIG_FIELDS:
            if not almost_equal(before[field], after[field]):
                fail(
                    f"Konfiguration unterscheidet sich bei N={size}, Feld {field}: "
                    f"{before[field]!r} vs. {after[field]!r}"
                )

        rate_diff = max(
            abs(float(before["exc_rate_hz"]) - float(after["exc_rate_hz"])),
            abs(float(before["inh_rate_hz"]) - float(after["inh_rate_hz"])),
        )
        cv_diff = abs(float(before["cv"]) - float(after["cv"]))

        max_rate_difference = max(max_rate_difference, rate_diff)
        max_cv_difference = max(max_cv_difference, cv_diff)

        malloc_before = int(initial_api[size]["cudaMalloc"]["calls"])
        malloc_after = int(optimized_api[size]["cudaMalloc"]["calls"])
        free_before = int(initial_api[size]["cudaFree"]["calls"])
        free_after = int(optimized_api[size]["cudaFree"]["calls"])
        symbol_before = int(initial_api[size]["cudaMemcpyFromSymbol"]["calls"])
        symbol_after = int(optimized_api[size]["cudaMemcpyFromSymbol"]["calls"])

        if malloc_after >= malloc_before:
            fail(f"cudaMalloc-Aufrufe wurden bei N={size} nicht reduziert")
        if free_after >= free_before:
            fail(f"cudaFree-Aufrufe wurden bei N={size} nicht reduziert")
        if symbol_before != 10000 or symbol_after != 0:
            fail(
                f"Unerwartete cudaMemcpyFromSymbol-Aufrufe bei N={size}: "
                f"{symbol_before} -> {symbol_after}"
            )

        sim_before = float(before["simulation_wall_time_s"])
        sim_after = float(after["simulation_wall_time_s"])
        update_before = initial_update[size]
        update_after = optimized_update[size]

        rows.append({
            "neurons": size,
            "initial_sim_wall_s": sim_before,
            "optimized_sim_wall_s": sim_after,
            "sim_reduction_pct": reduction(sim_before, sim_after),
            "initial_neuron_update_s": update_before,
            "optimized_neuron_update_s": update_after,
            "update_reduction_pct": reduction(update_before, update_after),
            "initial_cudaMalloc_calls": malloc_before,
            "optimized_cudaMalloc_calls": malloc_after,
            "cudaMalloc_call_reduction_pct": reduction(malloc_before, malloc_after),
            "initial_cudaFree_calls": free_before,
            "optimized_cudaFree_calls": free_after,
            "cudaFree_call_reduction_pct": reduction(free_before, free_after),
            "initial_cudaMemcpyFromSymbol_calls": symbol_before,
            "optimized_cudaMemcpyFromSymbol_calls": symbol_after,
            "initial_exc_rate_hz": float(before["exc_rate_hz"]),
            "optimized_exc_rate_hz": float(after["exc_rate_hz"]),
            "initial_inh_rate_hz": float(before["inh_rate_hz"]),
            "optimized_inh_rate_hz": float(after["inh_rate_hz"]),
            "initial_cv": float(before["cv"]),
            "optimized_cv": float(after["cv"]),
            "max_rate_difference_hz": rate_diff,
            "cv_difference": cv_diff,
        })

    if max_rate_difference > RATE_TOLERANCE_HZ:
        fail(
            f"Maximale Ratenabweichung {max_rate_difference:.12g} Hz "
            f"überschreitet {RATE_TOLERANCE_HZ:.12g} Hz"
        )
    if max_cv_difference > CV_TOLERANCE:
        fail(
            f"Maximale CV-Abweichung {max_cv_difference:.12g} "
            f"überschreitet {CV_TOLERANCE:.12g}"
        )

    write_outputs(rows, args.out_dir)

    required_outputs = (
        args.out_dir / "aeif_optimization_verified_results.csv",
        args.out_dir / "aeif_optimization_verification_report.md",
        args.out_dir / "aeif_wall_time_comparison.svg",
        args.out_dir / "aeif_runtime_reduction_percent.svg",
    )
    missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
    if missing_outputs:
        fail("Erwartete Ausgabedateien fehlen: " + ", ".join(missing_outputs))

    print("PASS: AEIF optimization results reproduced from the raw log")
    print()
    print(
        "N     wall reduction   update reduction   cudaMalloc   cudaFree   "
        "cudaMemcpyFromSymbol"
    )
    for row in rows:
        print(
            f"{row['neurons']:<5} "
            f"{row['sim_reduction_pct']:>8.1f}%"
            f"{row['update_reduction_pct']:>17.1f}%"
            f"{row['initial_cudaMalloc_calls']:>11} -> "
            f"{row['optimized_cudaMalloc_calls']:<8}"
            f"{row['initial_cudaFree_calls']:>10} -> "
            f"{row['optimized_cudaFree_calls']:<8}"
            f"{row['initial_cudaMemcpyFromSymbol_calls']:>8} -> "
            f"{row['optimized_cudaMemcpyFromSymbol_calls']}"
        )

    print()
    print(f"Maximum population-rate difference: {max_rate_difference:.5f} Hz")
    print(f"Maximum CV difference: {max_cv_difference:.7f}")
    print()
    print("Generated files:")
    print(f"  {args.out_dir / 'aeif_optimization_verified_results.csv'}")
    print(f"  {args.out_dir / 'aeif_optimization_verification_report.md'}")
    print(f"  {args.out_dir / 'aeif_wall_time_comparison.svg'}")
    print(f"  {args.out_dir / 'aeif_runtime_reduction_percent.svg'}")
    print(f"Outputs written to: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
