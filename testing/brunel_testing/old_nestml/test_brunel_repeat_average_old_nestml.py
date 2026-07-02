#!/usr/bin/env python3

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_nestgpu_vars_script(impl):
    if impl == "builtin":
        return "/p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh"

    if impl == "old_nestml":
        return "/p/project1/cslns/natouf1/nest-gpu_old_nestml/install_old_nestml/bin/nestgpu_vars.sh"

    raise ValueError(impl)


def run_worker(compare_script, family, impl, n_neurons, sim_time, run_outdir):
    nestgpu_vars = get_nestgpu_vars_script(impl)

    run_outdir.mkdir(parents=True, exist_ok=True)

    worker_cmd = [
        sys.executable,
        str(compare_script),
        "--worker",
        "--family",
        family,
        "--impl",
        impl,
        "--n-neurons",
        str(n_neurons),
        "--sim-time",
        str(sim_time),
        "--outdir",
        str(run_outdir),
    ]

    worker_cmd_str = " ".join(shlex.quote(str(x)) for x in worker_cmd)

    shell_cmd = f"""
source {shlex.quote(nestgpu_vars)}

export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${{LD_LIBRARY_PATH}}"

echo "Running repeated worker:"
echo "  family  = {family}"
echo "  impl    = {impl}"
echo "  neurons = {n_neurons}"
echo "  outdir  = {run_outdir}"
echo "Using nestgpu_vars: {nestgpu_vars}"
echo "NESTGPU_LIB=${{NESTGPU_LIB}}"
echo "PYTHONPATH=${{PYTHONPATH}}"

{worker_cmd_str}
"""

    subprocess.run(["bash", "-lc", shell_cmd], check=True)


def load_stats(run_outdir, family, impl):
    path = Path(run_outdir) / f"{family}_{impl}_stats.json"

    with open(path) as f:
        stats = json.load(f)

    return stats


def safe_float(value):
    if value is None:
        return None

    value = float(value)

    if np.isnan(value):
        return None

    return value


def mean_std(values):
    values = [safe_float(v) for v in values]
    values = [v for v in values if v is not None]

    if len(values) == 0:
        return None, None

    mean = float(np.mean(values))

    if len(values) == 1:
        std = 0.0
    else:
        std = float(np.std(values, ddof=1))

    return mean, std


def write_raw_csv(rows, csv_file):
    fieldnames = [
        "repeat",
        "family",
        "impl",
        "requested_neurons",
        "actual_neurons",
        "n_exc",
        "n_inh",
        "ce",
        "ci",
        "neural_activity_simulation_time_ms",
        "building_time_s",
        "simulation_wall_time_s",
        "total_wall_time_s",
        "building_ms_per_neuron",
        "simulation_ms_per_neuron",
        "total_ms_per_neuron",
        "exc_rate_hz",
        "inh_rate_hz",
        "exc_rate_hz_per_neuron_count",
        "inh_rate_hz_per_neuron_count",
        "cv",
        "w_ex",
        "w_in",
        "poiss_rate",
        "poiss_weight",
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def aggregate_rows(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["family"],
            row["impl"],
            row["requested_neurons"],
            row["actual_neurons"],
            row["n_exc"],
            row["n_inh"],
            row["ce"],
            row["ci"],
        )

        groups[key].append(row)

    summary_rows = []

    metrics = [
        "building_time_s",
        "simulation_wall_time_s",
        "total_wall_time_s",
        "building_ms_per_neuron",
        "simulation_ms_per_neuron",
        "total_ms_per_neuron",
        "exc_rate_hz",
        "inh_rate_hz",
        "exc_rate_hz_per_neuron_count",
        "inh_rate_hz_per_neuron_count",
        "cv",
    ]

    for key, group_rows in groups.items():
        (
            family,
            impl,
            requested_neurons,
            actual_neurons,
            n_exc,
            n_inh,
            ce,
            ci,
        ) = key

        out = {
            "family": family,
            "impl": impl,
            "requested_neurons": requested_neurons,
            "actual_neurons": actual_neurons,
            "n_exc": n_exc,
            "n_inh": n_inh,
            "ce": ce,
            "ci": ci,
            "repeats": len(group_rows),
        }

        for metric in metrics:
            values = [r.get(metric) for r in group_rows]
            mean, std = mean_std(values)

            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std

        summary_rows.append(out)

    summary_rows.sort(
        key=lambda r: (
            r["family"],
            r["impl"],
            int(r["actual_neurons"]),
        )
    )

    return summary_rows


def write_summary_csv(summary_rows, csv_file):
    fieldnames = [
        "family",
        "impl",
        "requested_neurons",
        "actual_neurons",
        "n_exc",
        "n_inh",
        "ce",
        "ci",
        "repeats",
        "building_time_s_mean",
        "building_time_s_std",
        "simulation_wall_time_s_mean",
        "simulation_wall_time_s_std",
        "total_wall_time_s_mean",
        "total_wall_time_s_std",
        "building_ms_per_neuron_mean",
        "building_ms_per_neuron_std",
        "simulation_ms_per_neuron_mean",
        "simulation_ms_per_neuron_std",
        "total_ms_per_neuron_mean",
        "total_ms_per_neuron_std",
        "exc_rate_hz_mean",
        "exc_rate_hz_std",
        "inh_rate_hz_mean",
        "inh_rate_hz_std",
        "exc_rate_hz_per_neuron_count_mean",
        "exc_rate_hz_per_neuron_count_std",
        "inh_rate_hz_per_neuron_count_mean",
        "inh_rate_hz_per_neuron_count_std",
        "cv_mean",
        "cv_std",
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def nice_impl_name(impl):
    if impl == "builtin":
        return "built-in"
    if impl == "old_nestml":
        return "old NESTML"
    return impl


def plot_summary(summary_rows, outdir):
    outdir = Path(outdir)

    # Separate plots per neuron model family:
    # one PNG for AEIF built-in vs old NESTML
    # one PNG for IAF built-in vs old NESTML
    families = sorted(set(r["family"] for r in summary_rows))
    impl_order = ["builtin", "old_nestml"]

    for family in families:
        family_rows = [r for r in summary_rows if r["family"] == family]
        neuron_counts = sorted(set(int(r["actual_neurons"]) for r in family_rows))

        lookup = {
            (r["impl"], int(r["actual_neurons"])): r
            for r in family_rows
        }

        labels = [f"N={n}" for n in neuron_counts]
        x = np.arange(len(neuron_counts))

        # 4 bars per neuron count:
        # builtin building, builtin simulation, old_nestml building, old_nestml simulation
        bar_labels = []
        bar_values = []
        bar_errors = []

        for impl in impl_order:
            build_values = []
            build_errors = []
            sim_values = []
            sim_errors = []

            for n in neuron_counts:
                row = lookup.get((impl, n))

                if row is None:
                    build_values.append(np.nan)
                    build_errors.append(0.0)
                    sim_values.append(np.nan)
                    sim_errors.append(0.0)
                    continue

                build_values.append(row["building_time_s_mean"])
                build_errors.append(row["building_time_s_std"] or 0.0)

                sim_values.append(row["simulation_wall_time_s_mean"])
                sim_errors.append(row["simulation_wall_time_s_std"] or 0.0)

            bar_labels.append(f"{nice_impl_name(impl)} building")
            bar_values.append(build_values)
            bar_errors.append(build_errors)

            bar_labels.append(f"{nice_impl_name(impl)} simulation")
            bar_values.append(sim_values)
            bar_errors.append(sim_errors)

        n_bars = len(bar_values)
        width = 0.8 / n_bars

        fig_width = max(10, len(neuron_counts) * 1.6)
        fig_height = 6

        plt.figure(figsize=(fig_width, fig_height))

        for i in range(n_bars):
            offset = (i - (n_bars - 1) / 2.0) * width

            plt.bar(
                x + offset,
                bar_values[i],
                width,
                yerr=bar_errors[i],
                capsize=3,
                label=bar_labels[i],
            )

        plt.ylabel("time [s]")
        plt.xlabel("number of neurons")
        plt.title(
            f"{family.upper()} Brunel repeated average timing "
            f"(built-in vs old NESTML)"
        )

        plt.xticks(x, labels)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plot_file = outdir / f"brunel_repeat_average_old_nestml_{family}_timing_summary.png"
        plt.savefig(plot_file, dpi=250)
        plt.close()

        print(f"Wrote plot to: {plot_file}")


def print_summary_table(summary_rows):
    print()
    print("=" * 120)
    print("Repeated Brunel average timing summary: built-in vs old NESTML")
    print("=" * 120)

    print(
        f"{'family':<8} "
        f"{'impl':<12} "
        f"{'N':>8} "
        f"{'rep':>5} "
        f"{'build[s]':>14} "
        f"{'sim[s]':>14} "
        f"{'build/N[ms]':>14} "
        f"{'sim/N[ms]':>14} "
        f"{'CV':>10}"
    )

    for r in summary_rows:
        cv = r["cv_mean"]
        cv_str = "nan" if cv is None else f"{cv:.4f}"

        print(
            f"{r['family']:<8} "
            f"{r['impl']:<12} "
            f"{r['actual_neurons']:>8} "
            f"{r['repeats']:>5} "
            f"{r['building_time_s_mean']:>14.8f} "
            f"{r['simulation_wall_time_s_mean']:>14.8f} "
            f"{r['building_ms_per_neuron_mean']:>14.8f} "
            f"{r['simulation_ms_per_neuron_mean']:>14.8f} "
            f"{cv_str:>10}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--neurons",
        nargs="+",
        type=int,
        default=[1000, 2000, 5000],
        help="Neuron counts to test.",
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=["iaf", "aeif"],
        default=["iaf", "aeif"],
        help="Neuron model families to test.",
    )

    parser.add_argument(
        "--impls",
        nargs="+",
        choices=["builtin", "old_nestml"],
        default=["builtin", "old_nestml"],
        help="Implementations to compare.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated runs per configuration.",
    )

    parser.add_argument(
        "--sim-time",
        type=float,
        default=1000.0,
        help="Biological simulation time in ms.",
    )

    parser.add_argument(
        "--outdir",
        default="brunel_repeat_average_old_nestml_out",
        help="Output directory.",
    )

    parser.add_argument(
        "--compare-script",
        default="/p/project1/cslns/natouf1/test_brunel_compare_overlay_old_nestml.py",
        help="Path to test_brunel_compare_overlay_old_nestml.py.",
    )

    args = parser.parse_args()

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    compare_script = Path(args.compare_script).resolve()

    rows = []

    for repeat in range(1, args.repeats + 1):
        print()
        print("#" * 100)
        print(f"Repeat {repeat}/{args.repeats}")
        print("#" * 100)

        for n_neurons in args.neurons:
            for family in args.families:
                for impl in args.impls:
                    run_outdir = (
                        base_outdir
                        / f"repeat_{repeat:02d}"
                        / f"{family}_{impl}_{n_neurons}"
                    )

                    print()
                    print("=" * 100)
                    print(
                        f"Running repeat={repeat}, "
                        f"family={family}, impl={impl}, neurons={n_neurons}"
                    )
                    print("=" * 100)

                    run_worker(
                        compare_script=compare_script,
                        family=family,
                        impl=impl,
                        n_neurons=n_neurons,
                        sim_time=args.sim_time,
                        run_outdir=run_outdir,
                    )

                    stats = load_stats(run_outdir, family, impl)
                    actual_neurons = stats["actual_neurons"]

                    stats["repeat"] = repeat

                    # Normalize timing values by the actual number of neurons.
                    stats["building_ms_per_neuron"] = (
                        stats["building_time_s"] * 1000.0 / actual_neurons
                    )

                    stats["simulation_ms_per_neuron"] = (
                        stats["simulation_wall_time_s"] * 1000.0 / actual_neurons
                    )

                    stats["total_ms_per_neuron"] = (
                        stats["total_wall_time_s"] * 1000.0 / actual_neurons
                    )

                    # Keep activity rates in the CSV/JSON for validation,
                    # but do not show them in the timing plots.
                    stats["exc_rate_hz_per_neuron_count"] = (
                        stats["exc_rate_hz"] / actual_neurons
                    )

                    stats["inh_rate_hz_per_neuron_count"] = (
                        stats["inh_rate_hz"] / actual_neurons
                    )

                    rows.append(stats)

                    print(
                        f"Result: family={family}, impl={impl}, "
                        f"N={actual_neurons}, "
                        f"build={stats['building_time_s']:.8f}s, "
                        f"sim={stats['simulation_wall_time_s']:.8f}s, "
                        f"build/N={stats['building_ms_per_neuron']:.8f}ms, "
                        f"sim/N={stats['simulation_ms_per_neuron']:.8f}ms, "
                        f"CV={stats['cv']}"
                    )

    raw_csv = base_outdir / "brunel_repeat_old_nestml_raw_results.csv"
    write_raw_csv(rows, raw_csv)

    summary_rows = aggregate_rows(rows)

    summary_csv = base_outdir / "brunel_repeat_old_nestml_average_summary.csv"
    write_summary_csv(summary_rows, summary_csv)

    plot_summary(summary_rows, base_outdir)
    print_summary_table(summary_rows)

    print()
    print(f"Wrote raw results to:     {raw_csv}")
    print(f"Wrote average summary to: {summary_csv}")


if __name__ == "__main__":
    main()
