import os
import numpy as np
import matplotlib.pyplot as plt

builtin_file = "/p/project1/cslns/natouf1/output_vm_aeif_builtin.txt"
odeint_file = "/p/project1/cslns/natouf1/output_vm_aeif_odeint.txt"
old_aeif_file = "/p/project1/cslns/natouf1/output_vm_aeif_old_nestml.txt"

plot_file = "/p/project1/cslns/natouf1/comparison_plot_aeif_solver_all.png"

for filename in [builtin_file, odeint_file, old_aeif_file]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")


b = np.loadtxt(builtin_file)
o = np.loadtxt(odeint_file)
old = np.loadtxt(old_aeif_file)


tb, Vb = b[:, 0], b[:, 1]
to, Vo = o[:, 0], o[:, 1]
told, Vold = old[:, 0], old[:, 1]


n = min(len(Vb), len(Vo), len(Vold))

if n == 0:
    raise RuntimeError("Comparison failed: one of the files is empty.")


tb = tb[:n]
to = to[:n]
told = told[:n]

Vb = Vb[:n]
Vo = Vo[:n]
Vold = Vold[:n]


if not np.allclose(tb, to):
    print("WARNING: time vectors builtin and odeint are not exactly identical.")
    print("max time difference builtin vs odeint =", np.max(np.abs(tb - to)))

if not np.allclose(tb, told):
    print("WARNING: time vectors builtin and old_aeif are not exactly identical.")
    print("max time difference builtin vs old_aeif =", np.max(np.abs(tb - told)))


diff_builtin_odeint = Vo - Vb
diff_builtin_old = Vold - Vb


rmse_builtin_odeint = np.sqrt(np.mean(diff_builtin_odeint ** 2))
max_abs_builtin_odeint = np.max(np.abs(diff_builtin_odeint))

rmse_builtin_old = np.sqrt(np.mean(diff_builtin_old ** 2))
max_abs_builtin_old = np.max(np.abs(diff_builtin_old))


print("AEIF solver comparison")
print("Compared points =", n)
print()
print("builtin vs odeint:")
print("RMSE [mV]    =", rmse_builtin_odeint)
print("MAX_ABS [mV] =", max_abs_builtin_odeint)
print()
print("builtin vs old_aeif:")
print("RMSE [mV]    =", rmse_builtin_old)
print("MAX_ABS [mV] =", max_abs_builtin_old)


plt.figure(figsize=(10, 6))

plt.plot(tb, Vb, label="built-in aeif_cond_alpha")
plt.plot(tb, Vo, "--", label="NESTML Odeint aeif_cond_alpha_alt")
plt.plot(tb, Vold, ":", label="old NESTML aeif_cond_alpha_alt")

plt.xlabel("time [ms]")
plt.ylabel("V_m [mV]")
plt.title("AEIF solver comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_file, dpi=200)
plt.close()

print()
print(f"Saved plot: {plot_file}")
