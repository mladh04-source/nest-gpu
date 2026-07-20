import os
import numpy as np
import nestgpu as ngpu
import matplotlib.pyplot as plt


# Configuration from environment
model_name = os.environ.get("MODEL_NAME", "aeif_cond_alpha")
outfile = os.environ.get("OUTFILE", "output_vm.txt")
plot_file = os.environ.get("PLOT_FILE", "plot.png")

plot_label_override = os.environ.get("PLOT_LABEL")


def get_plot_identity(model_name, outfile):
    if plot_label_override:
        return plot_label_override, "tab:orange"

    name = model_name.lower()
    output_name = outfile.lower()

    if name == "aeif_cond_alpha":
        return "Native AEIF", "tab:blue"

    if "old" in name or "old" in output_name:
        return "Legacy generated AEIF", "tab:green"

    return "Current Odeint AEIF", "tab:orange"


plot_label, plot_color = get_plot_identity(model_name, outfile)


# Setup
ngpu.SetTimeResolution(0.1)
ngpu.SetRandomSeed(123)

print("MODEL_NAME =", model_name)
print("OUTFILE =", outfile)
print("PLOT_FILE =", plot_file)


# Create neuron
neuron = ngpu.Create(model_name, 1)

try:
    print("ScalVars =", ngpu.GetScalVarNames(int(neuron[0])))
except Exception as e:
    print("Could not read scalar vars:", e)

try:
    print("IntVars =", ngpu.GetIntVarNames(int(neuron[0])))
except Exception as e:
    print("Could not read int vars:", e)

try:
    print("Params =", ngpu.GetScalParamNames(int(neuron[0])))
except Exception as e:
    print("Could not read scalar params:", e)


# Set parameters
# Only parameters that exist in the selected model are set.
#
# very Important:
# Built-in aeif_cond_alpha and NESTML aeif_cond_alpha_alt_neuron_nestml use different names for some equivalent parameters.
#
# Built-in names: tau_syn_ex, tau_syn_in, E_rev_ex, E_rev_in, t_ref
#
# NESTML names: tau_syn_exc, tau_syn_inh, E_exc, E_inh, refr_T

candidate_params = {
    "I_e": 200.0,
    "E_L": -70.6,
    "V_reset": -60.0,
    "V_th": -50.4,
    "V_peak": 0.0,
    "C_m": 281.0,
    "g_L": 30.0,
    "a": 4.0,
    "b": 80.5,
    "Delta_T": 2.0,
    "tau_w": 144.0,

    # NESTML aeif_cond_alpha_alt_neuron_nestml parameter names
    "tau_syn_exc": 0.2,
    "tau_syn_inh": 2.0,
    "E_exc": 0.0,
    "E_inh": -85.0,
    "refr_T": 2.0,

    # Built-in aeif_cond_alpha parameter names
    "tau_syn_ex": 0.2,
    "tau_syn_in": 2.0,
    "E_rev_ex": 0.0,
    "E_rev_in": -85.0,
    "t_ref": 2.0,

    # NESTML extra input current parameter
    "I_stim": 0.0,
}

available_params = []
try:
    available_params = ngpu.GetScalParamNames(int(neuron[0]))
except Exception:
    available_params = []

filtered_params = {
    k: v for k, v in candidate_params.items()
    if k in available_params
}

print("Setting params:", filtered_params)

if filtered_params:
    ngpu.SetStatus(neuron, filtered_params)

for k in filtered_params:
    try:
        val = ngpu.GetNeuronStatus(int(neuron[0]), k)
        print(f"PARAM {k} = {val}")
    except Exception:
        pass


# Spike generator input
spike = ngpu.Create("spike_generator")

spike_times = [10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
n_spikes = len(spike_times)

ngpu.SetStatus(spike, {"spike_times": spike_times})

conn_spec = {"rule": "all_to_all"}

syn_spec_ex = {
    "receptor": 0,
    "weight": 100.0,
    "delay": 1.0,
}

ngpu.Connect(spike, neuron, conn_spec, syn_spec_ex)


# Choose recordable variable automatically
record_var = None
scal_vars = ngpu.GetScalVarNames(int(neuron[0]))

print("Available record vars:", scal_vars)

if "V_m" in scal_vars:
    record_var = "V_m"
elif "V_m_rel" in scal_vars:
    record_var = "V_m_rel"
else:
    raise RuntimeError("Neither V_m nor V_m_rel available.")

print("Recording variable:", record_var)

record = ngpu.CreateRecord("", [record_var], [neuron[0]], [0])


# Simulate
sim_time = 500.0
ngpu.Simulate(sim_time)


# Read recorded data
data = ngpu.GetRecordData(record)

t = np.array([row[0] for row in data], dtype=float)
V_m = np.array([row[1] for row in data], dtype=float)

np.savetxt(outfile, np.column_stack((t, V_m)))

print("Saved:", outfile)
print("Recorded points:", len(t))
print("Configured spike count:", n_spikes)


# Optional spike count diagnostics
try:
    int_vars = ngpu.GetIntVarNames(int(neuron[0]))
    print("IntVars =", int_vars)

    if "spike_count" in int_vars:
        spike_count = ngpu.GetNeuronStatus(int(neuron[0]), "spike_count")
        print("Spike count =", spike_count)

except Exception as e:
    print("Could not read spike count:", e)


# Plot single model output
plt.figure(figsize=(10, 6))
plt.plot(t, V_m, color=plot_color, label=plot_label)

for ts in spike_times:
    plt.axvline(ts, linestyle=":", linewidth=0.8)

plt.xlabel("Time [ms]")
plt.ylabel("Membrane potential $V_m$ [mV]")
plt.title(plot_label)
plt.legend()
plt.tight_layout()
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()

print("Saved plot:", plot_file)
