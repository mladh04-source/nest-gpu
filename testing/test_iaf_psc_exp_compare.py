import os
import sys
import numpy as np
import nestgpu as ngpu
import matplotlib.pyplot as plt

# Configuration
tolerance = 1e-4

# Output file for recorded membrane potential
outfile = os.environ.get("OUTFILE", "output_vm.txt")

# reference file (for RMSE comparison)
# If not, the script will only simulate and save/plot NEST GPU output.
reference_file = os.environ.get("REFERENCE_FILE", "")

# Plot output
plot_file = os.environ.get("PLOT_FILE", "iaf_psc_exp_plot.png")

# Setup NEST GPU
ngpu.SetTimeResolution(0.1)
ngpu.SetRandomSeed(123)

# Create neuron
neuron = ngpu.Create("iaf_psc_exp", 1)


#spike-generator input.
ngpu.SetStatus(neuron, {
    "I_e": 1500.0,
    "tau_m": 10.0,
    "C_m": 250.0,
    "E_L": -70.0
})


# spike generator input
spike = ngpu.Create("spike_generator")

# Example spike times in ms
spike_times = [10.0, 400.0]
n_spikes = len(spike_times)

# Set spike times
ngpu.SetStatus(spike, {"spike_times": spike_times})

# Connection settings
conn_spec = {"rule": "all_to_all"}
syn_spec_ex = {
    "receptor": 0,
    "weight": 100.0,   
    "delay": 1.0
}

ngpu.Connect(spike, neuron, conn_spec, syn_spec_ex)

# Record membrane potential
record = ngpu.CreateRecord("", ["V_m_rel"], [neuron[0]], [0])

# Simulate
sim_time = 500.0
ngpu.Simulate(sim_time)

# Read recorded data
data = ngpu.GetRecordData(record)
t = [row[0] for row in data]
V_m = [row[1] for row in data]

# Save NEST GPU trace
np.savetxt(outfile, np.column_stack((t, V_m)))
print("Saved:", outfile)
print("OUTFILE env =", os.environ.get("OUTFILE"))
print("Recorded points:", len(t))
print("Configured spike count:", n_spikes)

#  reference comparison
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
        rmse = np.sqrt(np.mean((np.array(V_m[:n]) - np.array(V_m1[:n]))**2))
        print("rmse :", rmse, " tolerance: ", tolerance)
       
# Plot
fig1 = plt.figure(1)
plt.plot(t, V_m, "r-", label="NEST GPU")

if have_reference:
    plt.plot(t1, V_m1, "b--", label="Reference")

plt.xlabel("time [ms]")
plt.ylabel("V_m_rel")
plt.legend()
plt.draw()
plt.savefig(plot_file)
plt.close()

print("Saved plot:", plot_file)
