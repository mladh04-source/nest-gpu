import numpy as np
import matplotlib.pyplot as plt

rk5_file    = "/p/project1/cslns/natouf1/output_vm_aeif_rk5.txt"
odeint_file = "/p/project1/cslns/natouf1/output_vm_aeif_odeint.txt"

r = np.loadtxt(rk5_file)
o = np.loadtxt(odeint_file)

tr, Vr = r[:, 0], r[:, 1]
to, Vo = o[:, 0], o[:, 1]

n = min(len(Vr), len(Vo))

tr = tr[:n]
Vr = Vr[:n]
Vo = Vo[:n]

rmse_rk5_odeint = np.sqrt(np.mean((Vr - Vo) ** 2))
max_abs = np.max(np.abs(Vr - Vo))

print("RMSE rk5 vs odeint =", rmse_rk5_odeint)
print("MAX_ABS rk5 vs odeint =", max_abs)

plt.figure(figsize=(10, 6))
plt.plot(tr, Vr, label="aeif_cond_alpha_alt_neuron_nestml rk5")
plt.plot(tr, Vo, "--", label="aeif_cond_alpha_alt_neuron_nestml odeint")

plt.xlabel("time (ms)")
plt.ylabel("V_m")
plt.title(
    "AEIF solver comparison\n"
    f"RMSE rk5-odeint = {rmse_rk5_odeint:.6g}, "
    f"MAX_ABS = {max_abs:.6g}"
)
plt.legend()
plt.tight_layout()
plt.savefig("/p/project1/cslns/natouf1/comparison_plot_aeif_solver.png", dpi=200)

print("Saved /p/project1/cslns/natouf1/comparison_plot_aeif_solver.png")
