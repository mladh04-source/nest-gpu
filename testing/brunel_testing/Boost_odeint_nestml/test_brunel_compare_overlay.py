#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_cv(spike_train):
    isi = np.diff(spike_train)
    if len(isi) == 0:
        return np.nan

    mean_isi = np.mean(isi)
    if mean_isi == 0:
        return np.nan

    return np.std(isi) / mean_isi


def get_spike_stats(ngpu, neurons, n_exc, n_inh, sim_time):
    spike_times = ngpu.GetRecSpikeTimes(neurons)

    exc_count = 0
    inh_count = 0
    cvs = []

    total_neurons = n_exc + n_inh

    for i_neur in range(total_neurons):
        spikes = spike_times[i_neur]

        if len(spikes) > 2:
            cv = compute_cv(spikes)
            if not np.isnan(cv):
                cvs.append(cv)

        if i_neur < n_exc:
            exc_count += len(spikes)
        else:
            inh_count += len(spikes)

    exc_rate = exc_count / (n_exc * sim_time) * 1000.0 if n_exc > 0 else 0.0
    inh_rate = inh_count / (n_inh * sim_time) * 1000.0 if n_inh > 0 else 0.0
    mean_cv = np.mean(cvs) if len(cvs) > 0 else np.nan

    return {
        "exc_spike_count": int(exc_count),
        "inh_spike_count": int(inh_count),
        "exc_rate_hz": float(exc_rate),
        "inh_rate_hz": float(inh_rate),
        "cv": None if np.isnan(mean_cv) else float(mean_cv),
    }


def get_param_names(ngpu, gid):
    try:
        return set(ngpu.GetScalParamNames(int(gid)))
    except Exception:
        return set()


def get_var_names(ngpu, gid):
    try:
        return set(ngpu.GetScalVarNames(int(gid)))
    except Exception:
        return set()


def available_names(ngpu, gid):
    return get_param_names(ngpu, gid) | get_var_names(ngpu, gid)


def set_status_if_available(ngpu, nodes, candidate_dict, label=""):
    names = available_names(ngpu, nodes[0])
    filtered = {k: v for k, v in candidate_dict.items() if k in names}

    print(f"{label} available names:", sorted(names))
    print(f"{label} setting:", filtered)

    if filtered:
        ngpu.SetStatus(nodes, filtered)

    return filtered


def choose_record_var(ngpu, neuron_id):
    names = available_names(ngpu, neuron_id)

    if "V_m" in names:
        return "V_m"

    if "V_m_rel" in names:
        return "V_m_rel"

    raise RuntimeError(f"Neither V_m nor V_m_rel found. Available names: {sorted(names)}")


def randomize_initial_v(ngpu, neurons, e_l, seed=1234):
    rng = np.random.default_rng(seed)

    n = len(neurons)
    values = rng.uniform(e_l - 2.0, e_l + 6.0, size=n)

    names = available_names(ngpu, neurons[0])

    if "V_m" in names:
        key = "V_m"
    elif "V_m_rel" in names:
        key = "V_m_rel"
    else:
        print("Warning: could not find V_m or V_m_rel for initial randomization.")
        return None

    for i in range(n):
        gid = neurons[i]
        ngpu.SetStatus([gid], {key: float(values[i])})

    return key


def build_iaf_network(ngpu, n_requested, sim_time):
    order = n_requested // 5

    n_exc = 4 * order
    n_inh = 1 * order
    n_neurons = n_exc + n_inh

    epsilon = 0.1
    ce = int(epsilon * n_exc)
    ci = int(epsilon * n_inh)

    g = 6.0
    w_ex = 10.0

    return {
        "order": order,
        "n_exc": n_exc,
        "n_inh": n_inh,
        "n_neurons": n_neurons,
        "ce": ce,
        "ci": ci,
        "g": g,
        "w_ex": w_ex,
        "sim_time": sim_time,
    }


def build_aeif_network(ngpu, n_requested, sim_time):
    order = n_requested // 5

    n_exc = 4 * order
    n_inh = 1 * order
    n_neurons = n_exc + n_inh

    epsilon = 0.1
    ce = int(epsilon * n_exc)
    ci = int(epsilon * n_inh)

    g = 8.0
    w_ex = 0.035

    return {
        "order": order,
        "n_exc": n_exc,
        "n_inh": n_inh,
        "n_neurons": n_neurons,
        "ce": ce,
        "ci": ci,
        "g": g,
        "w_ex": w_ex,
        "sim_time": sim_time,
    }


def create_iaf_model(ngpu, impl, cfg):
    n_neurons = cfg["n_neurons"]

    if impl == "builtin":
        model_name = "iaf_psc_exp"
    elif impl == "nestml":
        model_name = "iaf_psc_exp_neuron_nestml"
    else:
        raise ValueError(impl)

    neuron = ngpu.Create(model_name, n_neurons)

    if impl == "builtin":
        params = {
            "C_m": 250.0,
            "tau_m": 10.0,
            "tau_syn_ex": 1.0,
            "tau_syn_in": 1.0,
            "t_ref": 2.0,
            "E_L": 0.0,
            "V_reset": 0.0,
            "V_m": 0.0,
            "V_m_rel": 0.0,
            "V_th": 20.0,
            "I_e": 0.0,
        }
    else:
        params = {
            "C_m": 250.0,
            "tau_m": 10.0,
            "tau_syn_exc": 1.0,
            "tau_syn_inh": 1.0,
            "refr_T": 2.0,
            "E_L": 0.0,
            "V_reset": 0.0,
            "V_m": 0.0,
            "V_th": 20.0,
            "I_e": 0.0,
            "I_stim": 0.0,
        }

    set_status_if_available(ngpu, neuron, params, label=f"IAF {impl}")
    randomize_initial_v(ngpu, neuron, e_l=0.0, seed=1234)

    return neuron


def create_aeif_model(ngpu, impl, cfg):
    n_neurons = cfg["n_neurons"]

    if impl == "builtin":
        model_name = "aeif_cond_alpha"
    elif impl == "nestml":
        model_name = "aeif_cond_alpha_alt_neuron_nestml"
    else:
        raise ValueError(impl)

    neuron = ngpu.Create(model_name, n_neurons)

    if impl == "builtin":
        params = {
            "C_m": 250.0,
            "g_L": 10.0,
            "E_L": -63.0,
            "V_reset": -65.0,
            "V_th": -50.0,
            "V_peak": -40.0,
            "a": 1.0,
            "b": 25.0,
            "Delta_T": 2.0,
            "tau_w": 150.0,
            "tau_syn_ex": 0.5,
            "tau_syn_in": 0.5,
            "E_ex": 0.0,
            "E_in": -85.0,
            "t_ref": 2.0,
            "I_e": 0.0,
        }
    else:
        params = {
            "C_m": 250.0,
            "g_L": 10.0,
            "E_L": -63.0,
            "V_reset": -65.0,
            "V_th": -50.0,
            "V_peak": -40.0,
            "a": 1.0,
            "b": 25.0,
            "Delta_T": 2.0,
            "tau_w": 150.0,
            "tau_syn_exc": 0.5,
            "tau_syn_inh": 0.5,
            "E_exc": 0.0,
            "E_inh": -85.0,
            "refr_T": 2.0,
            "I_e": 0.0,
            "I_stim": 0.0,
        }

    set_status_if_available(ngpu, neuron, params, label=f"AEIF {impl}")
    randomize_initial_v(ngpu, neuron, e_l=-63.0, seed=1234)

    return neuron


def connect_brunel_network(ngpu, family, impl, neuron, cfg):
    n_exc = cfg["n_exc"]
    n_neurons = cfg["n_neurons"]
    ce = cfg["ce"]
    ci = cfg["ci"]

    exc_neuron = neuron[0:n_exc]
    inh_neuron = neuron[n_exc:n_neurons]

    w_ex = cfg["w_ex"]
    g = cfg["g"]

    if family == "iaf":
        if impl == "builtin":
            w_in = -g * w_ex
            poiss_weight = 37.0
        else:
            w_in = g * w_ex
            poiss_weight = 97.0

        poiss_rate = 4800.0

    elif family == "aeif":
        w_in = g * w_ex
        poiss_weight = 9.1
        poiss_rate = 200.0

    else:
        raise ValueError(family)

    poiss_delay = 1.5

    exc_conn_dict = {"rule": "fixed_indegree", "indegree": ce}
    exc_syn_dict = {"weight": w_ex, "delay": 1.5, "receptor": 0}
    ngpu.Connect(exc_neuron, neuron, exc_conn_dict, exc_syn_dict)

    inh_conn_dict = {"rule": "fixed_indegree", "indegree": ci}
    inh_syn_dict = {"weight": w_in, "delay": 1.5, "receptor": 1}
    ngpu.Connect(inh_neuron, neuron, inh_conn_dict, inh_syn_dict)

    pg = ngpu.Create("poisson_generator")
    ngpu.SetStatus(pg, "rate", poiss_rate)

    pg_conn_dict = {"rule": "all_to_all"}
    pg_syn_dict = {"weight": poiss_weight, "delay": poiss_delay, "receptor": 0}
    ngpu.Connect(pg, neuron, pg_conn_dict, pg_syn_dict)

    return {
        "w_ex": w_ex,
        "w_in": w_in,
        "poiss_rate": poiss_rate,
        "poiss_weight": poiss_weight,
        "poiss_delay": poiss_delay,
    }


def run_worker(args):
    import nestgpu as ngpu

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.family == "all":
        raise RuntimeError("--worker needs --family iaf or --family aeif, not all.")

    ngpu.SetKernelStatus("rnd_seed", 1234)
    ngpu.SetTimeResolution(0.1)

    build_start = time.perf_counter()

    if args.family == "iaf":
        cfg = build_iaf_network(ngpu, args.n_neurons, args.sim_time)
        neuron = create_iaf_model(ngpu, args.impl, cfg)
    elif args.family == "aeif":
        cfg = build_aeif_network(ngpu, args.n_neurons, args.sim_time)
        neuron = create_aeif_model(ngpu, args.impl, cfg)
    else:
        raise ValueError(args.family)

    conn_info = connect_brunel_network(ngpu, args.family, args.impl, neuron, cfg)

    ngpu.ActivateRecSpikeTimes(neuron, 2000)

    n_neurons = cfg["n_neurons"]

    record_neurons = [
        neuron[min(37, n_neurons - 1)],
        neuron[min(100, n_neurons - 1)],
        neuron[n_neurons - 1],
    ]

    record_var = choose_record_var(ngpu, neuron[0])

    filename = str(outdir / f"{args.family}_{args.impl}_record.dat")

    record = ngpu.CreateRecord(
        filename,
        [record_var, record_var, record_var],
        record_neurons,
        [0, 0, 0],
    )

    build_end = time.perf_counter()

    sim_start = time.perf_counter()
    ngpu.Simulate(args.sim_time)
    sim_end = time.perf_counter()

    data_list = ngpu.GetRecordData(record)
    data = np.array(data_list, dtype=float)

    t = data[:, 0]
    v = data[:, 1:4]

    trace_file = outdir / f"{args.family}_{args.impl}_traces.npz"

    np.savez(
        trace_file,
        t=t,
        v=v,
        record_var=np.array(record_var),
        record_neurons=np.array([int(x) for x in record_neurons]),
    )

    spike_stats = get_spike_stats(
        ngpu,
        neuron,
        cfg["n_exc"],
        cfg["n_inh"],
        args.sim_time,
    )

    stats = {
        "family": args.family,
        "impl": args.impl,
        "requested_neurons": args.n_neurons,
        "actual_neurons": cfg["n_neurons"],
        "n_exc": cfg["n_exc"],
        "n_inh": cfg["n_inh"],
        "ce": cfg["ce"],
        "ci": cfg["ci"],
        "record_var": record_var,
        "record_neurons": [int(x) for x in record_neurons],
        "neural_activity_simulation_time_ms": args.sim_time,
        "building_time_s": build_end - build_start,
        "simulation_wall_time_s": sim_end - sim_start,
        "total_wall_time_s": sim_end - build_start,
        **conn_info,
        **spike_stats,
    }

    stats_file = outdir / f"{args.family}_{args.impl}_stats.json"

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


def load_trace(outdir, family, impl):
    path = Path(outdir) / f"{family}_{impl}_traces.npz"
    data = np.load(path, allow_pickle=True)

    t = data["t"]
    v = data["v"]
    record_var = str(data["record_var"].item())
    record_neurons = data["record_neurons"]

    return t, v, record_var, record_neurons


def load_stats(outdir, family, impl):
    path = Path(outdir) / f"{family}_{impl}_stats.json"
    with open(path) as f:
        return json.load(f)


def plot_overlay(outdir, family):
    outdir = Path(outdir)

    t_builtin, v_builtin, var_builtin, rec_builtin = load_trace(outdir, family, "builtin")
    t_nestml, v_nestml, var_nestml, rec_nestml = load_trace(outdir, family, "nestml")

    model_family = family.upper()
    native_label = f"Native {model_family}"
    current_label = f"Current Odeint {model_family}"

    for i in range(3):
        plt.figure(figsize=(11, 5))

        plt.plot(
            t_builtin,
            v_builtin[:, i],
            label=native_label,
            color="tab:blue",
            linewidth=1.2,
        )

        plt.plot(
            t_nestml,
            v_nestml[:, i],
            label=current_label,
            color="tab:orange",
            linewidth=1.0,
            alpha=0.85,
        )

        plt.xlabel("Time [ms]")
        plt.ylabel("Membrane potential [mV]")
        plt.title(
            f"{model_family} comparison: Native vs Current Odeint "
            f"(trace {i + 1})"
        )
        plt.legend()
        plt.tight_layout()

        outfile = outdir / f"{family}_native_vs_current_odeint_trace_{i + 1}.png"
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Wrote {outfile}")


def print_summary(outdir, family):
    builtin = load_stats(outdir, family, "builtin")
    nestml = load_stats(outdir, family, "nestml")

    print()
    print(f"===== {family.upper()} comparison summary =====")
    print(
        f"{'impl':<10} {'neurons':>8} {'build[s]':>12} "
        f"{'sim wall[s]':>12} {'activity[ms]':>14} "
        f"{'exc Hz':>10} {'inh Hz':>10} {'CV':>10}"
    )

    for s in [builtin, nestml]:
        cv = s["cv"]
        cv_str = "nan" if cv is None else f"{cv:.4f}"

        print(
            f"{s['impl']:<10} "
            f"{s['actual_neurons']:>8} "
            f"{s['building_time_s']:>12.4f} "
            f"{s['simulation_wall_time_s']:>12.4f} "
            f"{s['neural_activity_simulation_time_ms']:>14.1f} "
            f"{s['exc_rate_hz']:>10.2f} "
            f"{s['inh_rate_hz']:>10.2f} "
            f"{cv_str:>10}"
        )


def get_nestgpu_vars_script(impl):
    if impl == "builtin":
        return "/p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh"

    if impl == "nestml":
        return "/p/project1/cslns/natouf1/nest-gpu/install/bin/nestgpu_vars.sh"

    raise ValueError(impl)


def run_parent(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    families = ["iaf", "aeif"] if args.family == "all" else [args.family]

    script_path = str(Path(__file__).resolve())

    for family in families:
        for impl in ["builtin", "nestml"]:
            nestgpu_vars = get_nestgpu_vars_script(impl)

            worker_cmd = [
                sys.executable,
                script_path,
                "--worker",
                "--family",
                family,
                "--impl",
                impl,
                "--n-neurons",
                str(args.n_neurons),
                "--sim-time",
                str(args.sim_time),
                "--outdir",
                str(outdir),
            ]

            worker_cmd_str = " ".join(shlex.quote(str(x)) for x in worker_cmd)

            shell_cmd = f"""
source {shlex.quote(nestgpu_vars)}

export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${{LD_LIBRARY_PATH}}"

echo "Running worker: family={family}, impl={impl}"
echo "Using nestgpu_vars: {nestgpu_vars}"
echo "NESTGPU_LIB=${{NESTGPU_LIB}}"
echo "PYTHONPATH=${{PYTHONPATH}}"

{worker_cmd_str}
"""

            print()
            print(f"Running {family} {impl} with {nestgpu_vars}")
            subprocess.run(["bash", "-lc", shell_cmd], check=True)

        plot_overlay(outdir, family)
        print_summary(outdir, family)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--family", choices=["iaf", "aeif", "all"], default="all")
    parser.add_argument("--impl", choices=["builtin", "nestml"], default=None)
    parser.add_argument("--n-neurons", type=int, default=5000)
    parser.add_argument("--sim-time", type=float, default=1000.0)
    parser.add_argument("--outdir", default="brunel_compare_out")

    args = parser.parse_args()

    if args.worker:
        if args.impl is None:
            raise RuntimeError("--worker requires --impl builtin or --impl nestml.")
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
