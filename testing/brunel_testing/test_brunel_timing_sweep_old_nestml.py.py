#!/usr/bin/env python3

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


def load_stats(outdir, family, impl):
    path = Path(outdir) / f"{family}_{impl}_stats.json"
    with open(path) as f:
        return json.load(f)


def get_nestgpu_vars_script(impl):
    if impl == "builtin":
        return "/p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh"

    if impl == "old_nestml":
        return "/p/project1/cslns/natouf1/nest-gpu_old_nestml/install_old_nestml/bin/nestgpu_vars.sh"

    raise ValueError(impl)


def worker_impl_name(impl):
    """
    The worker script only knows 'builtin' and 'nestml'.
    For old_nestml, we still pass 'nestml' to the worker,
    but we source the old_nestml NEST-GPU installation before running it.
    so we use the old_nestml code for the comparison but it has the name nestml
    """
    if impl == "builtin":
        return "builtin"

    if impl == "old_nestml":
        return "nestml"

    raise ValueError(impl)


def run_worker(compare_script, family, impl, n_neurons, sim_time, run_outdir):
    nestgpu_vars = get_nestgpu_vars_script(impl)
    worker_impl = worker_impl_name(impl)

    worker_cmd = [
        sys.executable,
        str(compare_script),
        "--worker",
        "--family",
        family,
        "--impl",
        worker_impl,
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

echo "Running timing worker: family={family}, impl={impl}, worker_impl={worker_impl}, neurons={n_neurons}"
echo "Using nestgpu_vars: {nestgpu_vars}"
echo "NESTGPU_LIB=${{NESTGPU_LIB}}"
echo "PYTHONPATH=${{PYTHONPATH}}"

{worker_cmd_str}
"""

    subprocess.run(["bash", "-lc", shell_cmd], check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--neurons",
        nargs="+",
        type=int,
        default=[1000, 2000, 5000],
        help="Neuron counts to test, for example: --neurons 1000 2000 5000",
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=["iaf", "aeif"],
        default=["iaf", "aeif"],
    )

    parser.add_argument("--sim-time", type=float, default=1000.0)
    parser.add_argument("--outdir", default="brunel_timing_old_nestml_out")
    parser.add_argument(
        "--compare-script",
        default="/p/project1/cslns/natouf1/test_brunel_compare_overlay.py",
        help="Path to the worker script, for example test_brunel_compare_overlay.py",
    )

    args = parser.parse_args()

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    compare_script = Path(args.compare_script).resolve()

    rows = []

    for n_neurons in args.neurons:
        print()
        print("=" * 95)
        print(f"Neuron count: {n_neurons}")
        print("=" * 95)

        print(
            f"{'family':<8} {'impl':<12} {'neurons':>8} "
            f"{'building[s]':>14} {'simulation[s]':>14} "
            f"{'activity[ms]':>14} {'exc Hz':>10} {'inh Hz':>10} {'CV':>10}"
        )

        for family in args.families:
            for impl in ["builtin", "old_nestml"]:
                run_outdir = base_outdir / f"{family}_{impl}_{n_neurons}"

                run_worker(
                    compare_script=compare_script,
                    family=family,
                    impl=impl,
                    n_neurons=n_neurons,
                    sim_time=args.sim_time,
                    run_outdir=run_outdir,
                )

                worker_impl = worker_impl_name(impl)

                # The worker writes files with impl name 'builtin' or 'nestml'.
                # For old_nestml, the stats file is therefore family_nestml_stats.json.
                stats = load_stats(run_outdir, family, worker_impl)

                # Rename the implementation in the summary CSV/output.
                stats["impl"] = impl

                rows.append(stats)

                cv = stats["cv"]
                cv_str = "nan" if cv is None else f"{cv:.4f}"

                print(
                    f"{family:<8} "
                    f"{impl:<12} "
                    f"{stats['actual_neurons']:>8} "
                    f"{stats['building_time_s']:>14.4f} "
                    f"{stats['simulation_wall_time_s']:>14.4f} "
                    f"{stats['neural_activity_simulation_time_ms']:>14.1f} "
                    f"{stats['exc_rate_hz']:>10.2f} "
                    f"{stats['inh_rate_hz']:>10.2f} "
                    f"{cv_str:>10}"
                )

    csv_file = base_outdir / "brunel_timing_summary.csv"

    fieldnames = [
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
        "exc_rate_hz",
        "inh_rate_hz",
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

    print()
    print(f"Wrote timing summary to: {csv_file}")


if __name__ == "__main__":
    main()
