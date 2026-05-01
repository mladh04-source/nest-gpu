import sys
import numpy as np
import nestgpu as ngpu
import matplotlib.pyplot as plt


def compute_cv(spike_train):
    isi = np.diff(spike_train)
    if len(isi) == 0:
        return np.nan
    mean_isi = np.mean(isi)
    std_isi = np.std(isi)
    if mean_isi == 0:
        return np.nan
    return std_isi / mean_isi


def compute_cv_for_neurons(spike_trains):
    cvs = []
    for spike_train in spike_trains:
        if len(spike_train) > 1:
            cv = compute_cv(spike_train)
            if not np.isnan(cv):
                cvs.append(cv)
    return np.mean(cvs) if len(cvs) > 0 else np.nan


def get_spike_times(neurons, NE, NI):
    spike_times = ngpu.GetRecSpikeTimes(neurons)

    e_data = []
    i_data = []
    e_count = 0
    i_count = 0
    cvs = []

    for i_neur in range(NE + NI):
        spikes = spike_times[i_neur]
        if len(spikes) > 1:
            cv = compute_cv(spikes)
            if not np.isnan(cv):
                cvs.append(cv)

        if len(spikes) != 0:
            if i_neur < NE:
                for t in spikes:
                    e_count += 1
                    e_data.append([i_neur, t])
            else:
                for t in spikes:
                    i_count += 1
                    i_data.append([i_neur, t])

    mean_cv = np.mean(cvs) if len(cvs) > 0 else np.nan
    return e_data, i_data, e_count, i_count, mean_cv


def raster_plot(e_st, i_st, output_prefix="brunel_aeif_builtin"):
    e_ids = np.array([x[0] for x in e_st]) if len(e_st) else np.array([])
    e_times = np.array([x[1] for x in e_st]) if len(e_st) else np.array([])
    i_ids = np.array([x[0] for x in i_st]) if len(i_st) else np.array([])
    i_times = np.array([x[1] for x in i_st]) if len(i_st) else np.array([])

    plt.figure(figsize=(12, 8))
    if len(e_times):
        plt.plot(e_times, e_ids, ".", label="exc")
    if len(i_times):
        plt.plot(i_times, i_ids, ".", label="inh")
    plt.xlabel("time [ms]")
    plt.ylabel("neuron ID")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_raster.png", dpi=200)
    plt.close()


def compute_stats(ecount, icount, NE, NI, CE, CI, sim_time):
    rate_ex = ecount / (NE * sim_time) * 1000.0 if NE > 0 else 0.0
    rate_in = icount / (NI * sim_time) * 1000.0 if NI > 0 else 0.0

    print("Balanced network simulation statistics:")
    print(f"Number of neurons : {NE + NI}")
    print(f"Excitatory indegree: {CE}")
    print(f"Inhibitory indegree: {CI}")
    print(f"Excitatory rate   : {rate_ex:.2f} Hz")
    print(f"Inhibitory rate   : {rate_in:.2f} Hz")

    return rate_ex, rate_in


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} n_neurons")
        sys.exit(1)

    order = int(sys.argv[1]) // 5

    print("Building AEIF built-in Brunel network ...")

    ngpu.SetKernelStatus("rnd_seed", 1234)

    NE = 4 * order
    NI = 1 * order
    n_neurons = NE + NI

    epsilon = 0.1
    CE = int(epsilon * NE)
    CI = int(epsilon * NI)

    g = 4.0
    Wex = 6.0
    Win = g * Wex

    sim_time = 200.0

    poiss_rate = 4200.0
    poiss_weight = 35.0
    poiss_delay = 1.5

    neuron = ngpu.Create("aeif_cond_alpha", n_neurons)
    exc_neuron = neuron[0:NE]
    inh_neuron = neuron[NE:n_neurons]

    available_params = ngpu.GetScalParamNames(int(neuron[0]))
    print("Available params:", available_params)

    candidate_params = {
        "C_m": 200.0,
        "g_L": 10.0,
        "E_L": -63.0,
        "V_reset": -65.0,
        "V_th": -50.0,
        "V_peak": -40.0,
        "a": 0.0,
        "b": 40.0,
        "Delta_T": 2.0,
        "tau_w": 500.0,
        "tau_syn_ex": 0.5,
        "tau_syn_in": 0.5,
        "E_ex": 0.0,
        "E_in": -85.0,
        "t_ref": 0.0,
        "I_e": 200.0
    }

    filtered_params = {k: v for k, v in candidate_params.items() if k in available_params}
    print("Setting params:", filtered_params)
    if filtered_params:
        ngpu.SetStatus(neuron, filtered_params)

    exc_conn_dict = {"rule": "fixed_indegree", "indegree": CE}
    exc_syn_dict = {"weight": Wex, "delay": 1.5, "receptor": 0}
    ngpu.Connect(exc_neuron, neuron, exc_conn_dict, exc_syn_dict)

    inh_conn_dict = {"rule": "fixed_indegree", "indegree": CI}
    inh_syn_dict = {"weight": Win, "delay": 1.5, "receptor": 1}
    ngpu.Connect(inh_neuron, neuron, inh_conn_dict, inh_syn_dict)

    pg = ngpu.Create("poisson_generator")
    ngpu.SetStatus(pg, "rate", poiss_rate)

    pg_conn_dict = {"rule": "all_to_all"}
    pg_syn_dict = {"weight": poiss_weight, "delay": poiss_delay, "receptor": 0}
    ngpu.Connect(pg, neuron, pg_conn_dict, pg_syn_dict)

    ngpu.ActivateRecSpikeTimes(neuron, 2000)

    filename = "test_brunel_aeif_builtin.dat"
    i_neuron_arr = [neuron[37], neuron[min(100, n_neurons - 1)], neuron[n_neurons - 1]]
    i_receptor_arr = [0, 0, 0]
    var_name_arr = ["V_m", "V_m", "V_m"]

    record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr, i_receptor_arr)

    ngpu.Simulate(sim_time)

    e_data, i_data, ecount, icount, cv = get_spike_times(neuron, NE, NI)
    raster_plot(e_data, i_data, output_prefix="brunel_aeif_builtin")

    rate_ex, rate_in = compute_stats(ecount, icount, NE, NI, CE, CI, sim_time)
    print(f"CV: {cv}")

    data_list = ngpu.GetRecordData(record)
    t = [row[0] for row in data_list]
    V1 = [row[1] for row in data_list]
    V2 = [row[2] for row in data_list]
    V3 = [row[3] for row in data_list]

    plt.figure()
    plt.plot(t, V1)
    plt.tight_layout()
    plt.savefig("brunel_aeif_builtin_v1.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, V2)
    plt.tight_layout()
    plt.savefig("brunel_aeif_builtin_v2.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, V3)
    plt.tight_layout()
    plt.savefig("brunel_aeif_builtin_v3.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
