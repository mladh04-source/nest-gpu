import os
import numpy as np
import matplotlib.pyplot as plt

builtin_file = "/p/project1/cslns/natouf1/output_vm_aeif_builtin.txt"
odeint_file  = "/p/project1/cslns/natouf1/output_vm_aeif_odeint.txt"
plot_file    = "/p/project1/cslns/natouf1/comparison_plot_aeif_solver.png"

if not os.path.exists(builtin_file):
    raise FileNotFoundError(f"Builtin file not found: {builtin_file}")

if not os.path.exists(odeint_file):
    raise FileNotFoundError(f"ODEINT file not found: {odeint_file}")

b = np.loadtxt(builtin_file)
o = np.loadtxt(odeint_file)

tb, Vb = b[:, 0], b[:, 1]
to, Vo = o[:, 0], o[:, 1]

n = min(len(Vb), len(Vo))
if n == 0:
    raise RuntimeError("Comparison failed: one of the files is empty.")

tb = tb[:n]
Vb = Vb[:n]
Vo = Vo[:n]

rmse_builtin_odeint = np.sqrt(np.mean((Vb - Vo) ** 2))
max_abs = np.max(np.abs(Vb - Vo))

print("RMSE builtin vs odeint =", rmse_builtin_odeint)
print("MAX_ABS builtin vs odeint =", max_abs)
print("Compared points =", n)

plt.figure(figsize=(10, 6))
plt.plot(tb, Vb, label="aeif_cond_alpha_alt builtin")
plt.plot(tb, Vo, "--", label="aeif_cond_alpha_alt_neuron_nestml odeint")

plt.xlabel("time (ms)")
plt.ylabel("V_m")
plt.title("AEIF solver comparison")
plt.legend()
plt.tight_layout()
plt.savefig(plot_file, dpi=200)

print(f"Saved {plot_file}")
