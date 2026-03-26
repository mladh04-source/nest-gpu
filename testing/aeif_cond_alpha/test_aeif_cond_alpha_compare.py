import os
import numpy as np
import nestgpu as ngpu
import matplotlib.pyplot as plt


# Configuration
tolerance = 1e-4

# Which model to simulate
model_name = os.environ.get("MODEL_NAME", "aeif_cond_alpha")

# Output file for recorded membrane potential
outfile = os.environ.get("OUTFILE", "output_vm.txt")

# Optional reference file for RMSE comparison
reference_file = os.environ.get("REFERENCE_FILE", "")

# Plot output
plot_file = os.environ.get("PLOT_FILE", "aeif_cond_alpha_plot.png")


# Setup NEST GPU
ngpu.SetTimeResolution(0.1)
ngpu.SetRandomSeed(123)

# Create neuron
neuron = ngpu.Create(model_name, 1)
i0 = int(neuron[0])

# Print available names for debugging
scal_vars = ngpu.GetScalVarNames(i0)
int_vars = ngpu.GetIntVarNames(i0)
params = ngpu.GetScalParamNames(i0)

print("MODEL_NAME =", model_name)
print("ScalVars =", scal_vars)
print("IntVars  =", int_vars)
print("Params   =", params)


# Set physical parameters for the model
common_params = {
    "C_m": 281.0,
    "refr_T": 2.0,
    "V_reset": -60.0,
    "g_L": 30.0,
    "E_L": -70.6,
    "a": 4.0,
    "b": 80.5,
    "Delta_T": 2.0,
    "tau_w": 144.0,
    "V_th": -50.4,
    "V_peak": 0.0,
    "E_exc": 0.0,
    "tau_syn_exc": 0.2,
    "E_inh": -85.0,
    "tau_syn_inh": 2.0,
    "I_e": 200.0
}

# Only set parameters that the current model actually supports
filtered_params = {k: v for k, v in common_params.items() if k in params}

print("Setting parameters:", filtered_params)
ngpu.SetStatus(neuron, filtered_params)


# Spike generator input
spike = ngpu.Create("spike_generator")

spike_times = [10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
n_spikes = len(spike_times)

ngpu.SetStatus(spike, {"spike_times": spike_times})

conn_spec = {"rule": "all_to_all"}
syn_spec_ex = {
    "receptor": 0,
    "weight": 100.0,
    "delay": 1.0
}

ngpu.Connect(spike, neuron, conn_spec, syn_spec_ex)


# Choose membrane potential variable automatically
if "V_m" in scal_vars:
    vm_name = "V_m"
elif "V_m_rel" in scal_vars:
    vm_name = "V_m_rel"
else:
    raise RuntimeError("Neither 'V_m' nor 'V_m_rel' found in scalar variables")

print("Recording variable:", vm_name)


# Record membrane potential
record = ngpu.CreateRecord("", [vm_name], [neuron[0]], [0])

# Simulate
sim_time = 500.0
ngpu.Simulate(sim_time)


# Read recorded data
data = ngpu.GetRecordData(record)
t = [row[0] for row in data]
V_m = [row[1] for row in data]

np.savetxt(outfile, np.column_stack((t, V_m)))
print("Saved:", outfile)
print("OUTFILE env =", os.environ.get("OUTFILE"))
print("Recorded points:", len(t))
print("Configured spike count:", n_spikes)


# Optional reference comparison
have_reference = False
t1 = []
V_m1 = []

if reference_file and os.path.exists(reference_file):
    ref_data = np.loadtxt(reference_file)
    t1 = [x[0] for x in ref_data]
    V_m1 = [x[1] for x in ref_data]
    have_reference = True
    print("Loaded reference file:", reference_file)
    print("Reference points:", len(t1))
else:
    if reference_file:
        print("WARNING: reference file not found:", reference_file)
    else:
        print("No reference file specified. RMSE comparison skipped.")


# RMSE calculation
if have_reference:
    n = min(len(V_m), len(V_m1))
    dV = [V_m[i] - V_m1[i] for i in range(n)]

    if len(dV) == 0:
        print("WARNING: dV is empty, RMSE could not be computed.")
    else:
        rmse = np.sqrt(np.mean((np.array(V_m[:n]) - np.array(V_m1[:n])) ** 2))
        print("rmse :", rmse, " tolerance: ", tolerance)


# Plot
fig1 = plt.figure(1)
plt.plot(t, V_m, "r-", label=model_name)

if have_reference:
    plt.plot(t1, V_m1, "b--", label="Reference")

plt.xlabel("time [ms]")
plt.ylabel(vm_name)
plt.legend()
plt.draw()
plt.savefig(plot_file)
plt.close()

print("Saved plot:", plot_file)
