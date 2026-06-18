import os
import numpy as np
import matplotlib.pyplot as plt

builtin_file = "/p/project1/cslns/natouf1/output_vm_aeif_builtin.txt"
odeint_file  = "/p/project1/cslns/natouf1/output_vm_aeif_odeint.txt"

plot_file = "/p/project1/cslns/natouf1/comparison_plot_aeif_solver_with_diff.png"
diff_file = "/p/project1/cslns/natouf1/difference_vm_aeif_builtin_vs_odeint.txt"

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
to = to[:n]
Vb = Vb[:n]
Vo = Vo[:n]

if not np.allclose(tb, to):
    print("WARNING: time vectors are not exactly identical.")
    print("max time difference =", np.max(np.abs(tb - to)))

diff = Vo - Vb

rmse = np.sqrt(np.mean(diff ** 2))
mae = np.mean(np.abs(diff))
max_abs = np.max(np.abs(diff))

np.savetxt(
    diff_file,
    np.column_stack((tb, diff)),
    header="time_ms Vm_odeint_minus_builtin_mV"
)

print("AEIF single neuron numerical comparison")
print("Compared points =", n)
print("RMSE [mV]      =", rmse)
print("MAE [mV]       =", mae)
print("MAX_ABS [mV]   =", max_abs)
print("Saved diff data:", diff_file)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

axes[0].plot(tb, Vb, label="built-in aeif_cond_alpha")
axes[0].plot(tb, Vo, "--", label="NESTML Odeint aeif_cond_alpha_alt")
axes[0].set_ylabel("V_m [mV]")
axes[0].set_title("AEIF single neuron solver comparison")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(tb, diff, label="diff = Odeint - built-in")
axes[1].axhline(0.0, linestyle=":", linewidth=1.0)
axes[1].set_xlabel("time [ms]")
axes[1].set_ylabel("diff [mV]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_file, dpi=200)
plt.close()

print("Saved plot:", plot_file)
