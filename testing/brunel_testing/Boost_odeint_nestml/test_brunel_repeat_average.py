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

    if impl == "nestml":
        return "/p/project1/cslns/natouf1/nest-gpu/install_numeric_odeint/bin/nestgpu_vars.sh"

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


def plot_summary(summary_rows, outdir):
    outdir = Path(outdir)

    families = sorted(set(r["family"] for r in summary_rows))
    impl_order = ["builtin", "nestml"]

    for family in families:
        family_rows = [r for r in summary_rows if r["family"] == family]

        neuron_counts = sorted(set(int(r["actual_neurons"]) for r in family_rows))

        lookup = {
            (r["impl"], int(r["actual_neurons"])): r
            for r in family_rows
        }

        labels = [f"N={n}" for n in neuron_counts]
        x = np.arange(len(neuron_counts))

        width = 0.20

        plt.figure(figsize=(max(10, len(neuron_counts) * 1.8), 6))

        for impl_index, impl in enumerate(impl_order):
            building = []
            building_err = []
            simulation = []
            simulation_err = []

            for n in neuron_counts:
                row = lookup.get((impl, n))

                if row is None:
                    building.append(np.nan)
                    building_err.append(0.0)
                    simulation.append(np.nan)
                    simulation_err.append(0.0)
                    continue

                building.append(row["building_ms_per_neuron_mean"])
                building_err.append(row["building_ms_per_neuron_std"] or 0.0)

                simulation.append(row["simulation_ms_per_neuron_mean"])
                simulation_err.append(row["simulation_ms_per_neuron_std"] or 0.0)

            if impl == "builtin":
                build_offset = -1.5 * width
                sim_offset = -0.5 * width
            else:
                build_offset = 0.5 * width
                sim_offset = 1.5 * width

            plt.bar(
                x + build_offset,
                building,
                width,
                yerr=building_err,
                capsize=3,
                label=f"{impl} build/N",
            )

            plt.bar(
                x + sim_offset,
                simulation,
                width,
                yerr=simulation_err,
                capsize=3,
                label=f"{impl} sim/N",
            )

        plt.ylabel("time per neuron [ms/neuron]")
        plt.xlabel("number of neurons")
        plt.title(f"{family.upper()} Brunel repeated average timing summary")

        plt.xticks(x, labels)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plot_file = outdir / f"brunel_repeat_average_{family}_timing_summary.png"
        plt.savefig(plot_file, dpi=250)
        plt.close()

        print(f"Wrote plot to: {plot_file}")

def print_summary_table(summary_rows):
    print()
    print("=" * 150)
    print("Repeated Brunel average summary")
    print("=" * 150)

    print(
        f"{'family':<8} "
        f"{'impl':<10} "
        f"{'N':>8} "
        f"{'rep':>5} "
        f"{'build/N':>14} "
        f"{'sim/N':>14} "
        f"{'exc/N':>14} "
        f"{'inh/N':>14} "
        f"{'CV':>10}"
    )

    for r in summary_rows:
        cv = r["cv_mean"]
        cv_str = "nan" if cv is None else f"{cv:.4f}"

        exc_n = r["exc_rate_hz_per_neuron_count_mean"]
        inh_n = r["inh_rate_hz_per_neuron_count_mean"]

        exc_n_str = "nan" if exc_n is None else f"{exc_n:.8f}"
        inh_n_str = "nan" if inh_n is None else f"{inh_n:.8f}"

        print(
            f"{r['family']:<8} "
            f"{r['impl']:<10} "
            f"{r['actual_neurons']:>8} "
            f"{r['repeats']:>5} "
            f"{r['building_ms_per_neuron_mean']:>14.8f} "
            f"{r['simulation_ms_per_neuron_mean']:>14.8f} "
            f"{exc_n_str:>14} "
            f"{inh_n_str:>14} "
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
        choices=["builtin", "nestml"],
        default=["builtin", "nestml"],
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
        default="brunel_repeat_average_out",
        help="Output directory.",
    )

    parser.add_argument(
        "--compare-script",
        default="/p/project1/cslns/natouf1/test_brunel_compare_overlay.py",
        help="Path to test_brunel_compare_overlay.py.",
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

                    # Normalize activity rates by the actual number of neurons.
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
                        f"build/N={stats['building_ms_per_neuron']:.8f}, "
                        f"sim/N={stats['simulation_ms_per_neuron']:.8f}, "
                        f"exc/N={stats['exc_rate_hz_per_neuron_count']:.8f}, "
                        f"inh/N={stats['inh_rate_hz_per_neuron_count']:.8f}, "
                        f"CV={stats['cv']}"
                    )

    raw_csv = base_outdir / "brunel_repeat_raw_results.csv"
    write_raw_csv(rows, raw_csv)

    summary_rows = aggregate_rows(rows)

    summary_csv = base_outdir / "brunel_repeat_average_summary.csv"
    write_summary_csv(summary_rows, summary_csv)

    plot_summary(summary_rows, base_outdir)
    print_summary_table(summary_rows)

    print()
    print(f"Wrote raw results to:     {raw_csv}")
    print(f"Wrote average summary to: {summary_csv}")


if __name__ == "__main__":
    main()
