import sys
from random import randrange

import matplotlib.pyplot as plt
import nestgpu as ngpu
import numpy as np


def compute_cv(spike_train):
    isi = np.diff(spike_train)
    if len(isi) == 0:
        return np.nan
    mean_isi = np.mean(isi)
    if mean_isi == 0:
        return np.nan
    std_isi = np.std(isi)
    return std_isi / mean_isi


def compute_cv_for_neurons(spike_trains):
    cvs = []
    for spike_train in spike_trains:
        cv = compute_cv(spike_train)
        if not np.isnan(cv):
            cvs.append(cv)
    return np.mean(cvs) if cvs else np.nan


def get_spike_times(neurons, n_exc, n_inh):
    spike_times = ngpu.GetRecSpikeTimes(neurons)

    exc_data = []
    inh_data = []
    exc_count = 0
    inh_count = 0
    all_trains = []

    total_neurons = n_exc + n_inh

    for i_neur in range(total_neurons):
        spikes = spike_times[i_neur]
        all_trains.append(spikes)

        if len(spikes) == 0:
            continue

        if i_neur < n_exc:
            for t in spikes:
                exc_count += 1
                exc_data.append([i_neur, t])
        else:
            for t in spikes:
                inh_count += 1
                inh_data.append([i_neur, t])

    cv_mean = compute_cv_for_neurons(all_trains)
    return exc_data, inh_data, exc_count, inh_count, cv_mean


def raster_plot(exc_spikes, inh_spikes, n_neurons, outname="current_odeint_iaf_raster.png"):
    plt.figure(figsize=(12, 7))

    if exc_spikes:
        exc_ids = [x[0] for x in exc_spikes]
        exc_times = [x[1] for x in exc_spikes]
        plt.plot(exc_times, exc_ids, ".", color="tab:blue", markersize=2, label="Excitatory")

    if inh_spikes:
        inh_ids = [x[0] for x in inh_spikes]
        inh_times = [x[1] for x in inh_spikes]
        plt.plot(inh_times, inh_ids, ".", color="tab:orange", markersize=2, label="Inhibitory")

    plt.xlabel("Time [ms]")
    plt.ylabel("Neuron ID")
    plt.title(f"Current Odeint IAF (N = {n_neurons})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    plt.close()


def save_voltage_trace(record, outprefix="current_odeint_iaf"):
    data_list = ngpu.GetRecordData(record)
    t = [row[0] for row in data_list]
    traces = []

    ncols = ngpu.GetRecordDataColumns(record)
    for col in range(1, ncols):
        traces.append([row[col] for row in data_list])

    for i, trace in enumerate(traces, start=1):
        np.savetxt(
            f"{outprefix}_vm_{i}.txt",
            np.column_stack((t, trace)),
        )
        plt.figure(figsize=(10, 5))
        plt.plot(t, trace)
        plt.xlabel("Time [ms]")
        plt.ylabel("Membrane potential $V_m$ [mV]")
        plt.title(f"Current Odeint IAF: example neuron {i}")
        plt.tight_layout()
        plt.savefig(
            f"{outprefix}_voltage_trace_{i}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def compute_stats(exc_count, inh_count, n_exc, n_inh, sim_time, ce, ci):
    rate_ex = exc_count / (n_exc * sim_time) * 1000.0 if n_exc > 0 else 0.0
    rate_in = inh_count / (n_inh * sim_time) * 1000.0 if n_inh > 0 else 0.0

    print("Balanced network simulation statistics:")
    print(f"Number of neurons : {n_exc + n_inh}")
    print(f"Excitatory indegree : {ce}")
    print(f"Inhibitory indegree : {ci}")
    print(f"Approx. excitatory synapses : {int(ce * (n_exc + n_inh))}")
    print(f"Approx. inhibitory synapses : {int(ci * (n_exc + n_inh))}")
    print(f"Excitatory rate : {rate_ex:.2f} Hz")
    print(f"Inhibitory rate : {rate_in:.2f} Hz")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} n_neurons")
        sys.exit(1)

    total_requested = int(sys.argv[1])
    order = total_requested // 5

    print("Building Brunel network for iaf_psc_exp_neuron_nestml ...")

    ngpu.SetKernelStatus("rnd_seed", 1234)
    ngpu.SetTimeResolution(0.1)

    n_exc = 4 * order
    n_inh = 1 * order
    n_neurons = n_exc + n_inh

    epsilon = 0.1
    ce = int(epsilon * n_exc)
    ci = int(epsilon * n_inh)

    g = 6.0
    w_ex = 10.0
    w_in = g * w_ex

    sim_time = 1000.0

    poiss_rate = 4800.0
    poiss_weight = 97.0
    poiss_delay = 1.5

    tau_syn_exc = 1.0
    tau_syn_inh = 1.0
    tau_m = 10.0
    c_m = 250.0
    e_l = 0.0
    v_reset = 0.0
    v_th = 20.0
    refr_t = 2.0
    i_e = 0.0

    neuron = ngpu.Create("iaf_psc_exp_neuron_nestml", n_neurons)
    exc_neuron = neuron[0:n_exc]
    inh_neuron = neuron[n_exc:n_neurons]

    ngpu.SetStatus(
        neuron,
        {
            "C_m": c_m,
            "tau_m": tau_m,
            "tau_syn_exc": tau_syn_exc,
            "tau_syn_inh": tau_syn_inh,
            "refr_T": refr_t,
            "E_L": e_l,
            "V_reset": v_reset,
            "V_m": e_l,
            "V_th": v_th,
            "I_e": i_e,
            "I_stim": 0.0,
        },
    )

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

    ngpu.ActivateRecSpikeTimes(neuron, 2000)

    filename = "brunel_iaf_multimeter.dat"
    i_neuron_arr = [
        neuron[37],
        neuron[randrange(n_neurons)],
        neuron[n_neurons - 1],
    ]
    i_receptor_arr = [0, 0, 0]
    var_name_arr = ["V_m", "V_m", "V_m"]

    record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr, i_receptor_arr)

    ngpu.Simulate(sim_time)

    exc_data, inh_data, exc_count, inh_count, cv = get_spike_times(neuron, n_exc, n_inh)

    raster_plot(
        exc_data,
        inh_data,
        n_neurons,
        outname="current_odeint_iaf_raster.png",
    )
    save_voltage_trace(record, outprefix="current_odeint_iaf")

    compute_stats(exc_count, inh_count, n_exc, n_inh, sim_time, ce, ci)
    print(f"CV : {cv:.4f}" if not np.isnan(cv) else "CV : nan")


if __name__ == "__main__":
    main()
