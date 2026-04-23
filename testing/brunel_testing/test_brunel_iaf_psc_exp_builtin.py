import nestgpu as ngpu
import numpy as np
import matplotlib.pyplot as plt


def compute_cv(spike_train):
    isi = np.diff(spike_train)
    if len(isi) == 0:
        return np.nan

    mean_isi = np.mean(isi)
    if mean_isi == 0:
        return np.nan

    std_isi = np.std(isi)
    return std_isi / mean_isi


def get_spike_times(neurons, n_exc, n_inh):
    spike_times = ngpu.GetRecSpikeTimes(neurons)

    e_data = []
    i_data = []
    e_count = 0
    i_count = 0
    cvs = []

    total = n_exc + n_inh
    for i_neur in range(total):
        spikes = spike_times[i_neur]

        if len(spikes) > 1:
            cv = compute_cv(spikes)
            if not np.isnan(cv):
                cvs.append(cv)

        if len(spikes) != 0:
            if i_neur < n_exc:
                for t in spikes:
                    e_count += 1
                    e_data.append([i_neur, t])
            else:
                for t in spikes:
                    i_count += 1
                    i_data.append([i_neur, t])

    mean_cv = np.mean(cvs) if len(cvs) > 0 else np.nan
    return e_data, i_data, e_count, i_count, mean_cv


def raster_plot(e_st, i_st):
    colors = ["#595289", "#af143c"]

    e_ids = np.array([x[0] for x in e_st]) if len(e_st) else np.array([])
    e_times = np.array([x[1] for x in e_st]) if len(e_st) else np.array([])
    i_ids = np.array([x[0] for x in i_st]) if len(i_st) else np.array([])
    i_times = np.array([x[1] for x in i_st]) if len(i_st) else np.array([])

    plt.figure(figsize=(12, 7))
    if len(e_times):
        plt.plot(e_times, e_ids, ".", color=colors[0], label="exc")
    if len(i_times):
        plt.plot(i_times, i_ids, ".", color=colors[1], label="inh")

    plt.xlabel("time [ms]")
    plt.ylabel("neuron ID")
    plt.legend()
    plt.tight_layout()
    plt.savefig("brunel_iaf_builtin_raster.png", dpi=300)
    plt.close()


def compute_stats(ecount, icount, n_exc, n_inh, sim_time, ce, ci, n_neurons):
    rate_ex = ecount / (n_exc * sim_time) * 1000.0 if n_exc > 0 else 0.0
    rate_in = icount / (n_inh * sim_time) * 1000.0 if n_inh > 0 else 0.0

    print("Balanced network simulation statistics:")
    print(f"Number of neurons : {n_neurons}")
    print(f"Excitatory indegree: {ce}")
    print(f"Inhibitory indegree: {ci}")
    print(f"Excitatory rate   : {rate_ex:.2f} Hz")
    print(f"Inhibitory rate   : {rate_in:.2f} Hz")


def choose_record_var(neuron_id):
    try:
        available_vars = ngpu.GetScalVarNames(int(neuron_id))
        print("Available scalar variables:", available_vars)
    except Exception as e:
        raise RuntimeError(f"Could not read scalar variables: {e}")

    if "V_m" in available_vars:
        return "V_m"
    if "V_m_rel" in available_vars:
        return "V_m_rel"

    raise RuntimeError(
        f"Neither 'V_m' nor 'V_m_rel' available. Found: {available_vars}"
    )


def set_params_if_available(nodes, candidate_params):
    try:
        neuron_id = int(nodes[0])
        available_params = ngpu.GetScalParamNames(neuron_id)
        print("Available parameters:", available_params)
    except Exception as e:
        raise RuntimeError(f"Could not read scalar parameter names: {e}")

    filtered = {k: v for k, v in candidate_params.items() if k in available_params}
    print("Setting parameters:", filtered)

    if filtered:
        ngpu.SetStatus(nodes, filtered)
    else:
        print("Warning: no matching parameters found to set.")


def main():
    print("Building Brunel network with built-in iaf_psc_exp ...")

    ngpu.SetKernelStatus("rnd_seed", 1234)

    order = 1000
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

    neuron = ngpu.Create("iaf_psc_exp", n_neurons)
    exc_neuron = neuron[0:n_exc]
    inh_neuron = neuron[n_exc:n_neurons]

    candidate_params = {
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
    set_params_if_available(neuron, candidate_params)

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

    filename = "test_brunel_iaf_builtin.dat"
    i_neuron_arr = [neuron[37], neuron[min(100, n_neurons - 1)], neuron[n_neurons - 1]]
    i_receptor_arr = [0, 0, 0]

    record_var = choose_record_var(neuron[0])
    print("Recording variable:", record_var)

    var_name_arr = [record_var, record_var, record_var]
    record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr, i_receptor_arr)

    ngpu.Simulate(sim_time)

    e_data, i_data, ecount, icount, cv = get_spike_times(neuron, n_exc, n_inh)
    raster_plot(e_data, i_data)
    compute_stats(ecount, icount, n_exc, n_inh, sim_time, ce, ci, n_neurons)
    print(f"CV: {cv}")

    data_list = ngpu.GetRecordData(record)
    t = [row[0] for row in data_list]
    v1 = [row[1] for row in data_list]
    v2 = [row[2] for row in data_list]
    v3 = [row[3] for row in data_list]

    plt.figure()
    plt.plot(t, v1)
    plt.xlabel("time [ms]")
    plt.ylabel(record_var)
    plt.tight_layout()
    plt.savefig("brunel_iaf_builtin_v1.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, v2)
    plt.xlabel("time [ms]")
    plt.ylabel(record_var)
    plt.tight_layout()
    plt.savefig("brunel_iaf_builtin_v2.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, v3)
    plt.xlabel("time [ms]")
    plt.ylabel(record_var)
    plt.tight_layout()
    plt.savefig("brunel_iaf_builtin_v3.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
