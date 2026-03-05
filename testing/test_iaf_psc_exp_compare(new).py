import nestgpu as ngpu
import numpy as np
import os

# ---- Setup ----
ngpu.SetTimeResolution(0.1)
ngpu.SetRandomSeed(123)

neuron = ngpu.Create("iaf_psc_exp", 1)

# explicitly set parameters
ngpu.SetStatus(neuron, {
    "I_e": 1500.0,
    "tau_m": 10.0,
    "C_m": 250.0,
    "E_L": -70.0
})

# Record membrane potential
record = ngpu.CreateRecord("", ["V_m_rel"], [neuron[0]], [0])

ngpu.Simulate(200.0)

data = ngpu.GetRecordData(record)
t = [row[0] for row in data]
V = [row[1] for row in data]

# Save
out = os.environ.get("OUTFILE", "output_vm.txt")
print("OUTFILE env =", os.environ.get("OUTFILE"))
np.savetxt(out, np.column_stack((t, V)))
print("Saved", out)
