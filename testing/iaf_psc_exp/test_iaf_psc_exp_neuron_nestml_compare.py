import os
import numpy as np
import nestgpu as ngpu
import matplotlib.pyplot as plt


# Configuration from environment
model_name = os.environ.get("MODEL_NAME", "iaf_psc_exp_neuron_nestml")
outfile = os.environ.get("OUTFILE", "output_vm.txt")
plot_file = os.environ.get("PLOT_FILE", "iaf_psc_exp_plot.png")

plot_label_override = os.environ.get("PLOT_LABEL")


def get_plot_identity(model_name, outfile):
    if plot_label_override:
        return plot_label_override, "tab:orange"

    name = model_name.lower()
    output_name = outfile.lower()

    if name == "iaf_psc_exp":
        return "Native IAF", "tab:blue"

    if "analytic" in output_name or "old" in output_name:
        return "Legacy generated IAF", "tab:green"

    return "Current Odeint IAF", "tab:orange"


plot_label, plot_color = get_plot_identity(model_name, outfile)


# Setup NEST GPU
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


# Parameters
#
# Important:
# NESTML iaf_psc_exp_neuron_nestml uses absolute V_m.
# Built-in iaf_psc_exp uses V_m_rel relative to E_L.
#
# Therefore:
#   NESTML threshold: V_th = -55.0
#   Built-in threshold: Theta_rel = V_th - E_L = 15.0
#   NESTML reset: V_reset = -70.0
#   Built-in reset: V_reset_rel = V_reset - E_L = 0.0

E_L_value = -70.0
V_th_value = -55.0
V_reset_value = -70.0

candidate_params = {
    # Common physical parameters
    "I_e": 200.0,
    "tau_m": 10.0,
    "C_m": 250.0,
    "E_L": E_L_value,

    # NESTML parameter names
    "V_th": V_th_value,
    "V_reset": V_reset_value,
    "refr_T": 2.0,
    "tau_syn_exc": 2.0,
    "tau_syn_inh": 2.0,
    "I_stim": 0.0,

    # Built-in iaf_psc_exp parameter names
    "Theta_rel": V_th_value - E_L_value,
    "V_reset_rel": V_reset_value - E_L_value,
    "t_ref": 2.0,
    "tau_ex": 2.0,
    "tau_in": 2.0,
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
V_raw = np.array([row[1] for row in data], dtype=float)


# Convert output to absolute V_m for fair comparison.
#
# Built-in iaf_psc_exp records V_m_rel.
# NESTML iaf_psc_exp_neuron_nestml records absolute V_m.

if record_var == "V_m_rel":
    V_out = V_raw + E_L_value
    output_label = "V_m_abs_from_V_m_rel"
else:
    V_out = V_raw
    output_label = "V_m_abs"

np.savetxt(outfile, np.column_stack((t, V_out)))

print("Saved:", outfile)
print("Saved variable:", output_label)
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
plt.plot(t, V_out, color=plot_color, label=plot_label)

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
