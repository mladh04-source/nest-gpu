import os
import numpy as np
import matplotlib.pyplot as plt


base_dir = os.environ.get("BASE_DIR", "/p/project1/cslns/natouf1")

analytic_file = os.path.join(base_dir, "output_vm_analytic.txt")
numeric_file  = os.path.join(base_dir, "output_vm_numeric.txt")
builtin_file  = os.path.join(base_dir, "output_vm_builtin.txt")

plot_file = os.path.join(base_dir, "comparison_plot_iaf_three_models.png")


for filename in [analytic_file, numeric_file, builtin_file]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")


analytic = np.loadtxt(analytic_file)
numeric  = np.loadtxt(numeric_file)
builtin  = np.loadtxt(builtin_file)


ta, Va = analytic[:, 0], analytic[:, 1]
tn, Vn = numeric[:, 0], numeric[:, 1]
tb, Vb = builtin[:, 0], builtin[:, 1]


n = min(len(Va), len(Vn), len(Vb))

if n == 0:
    raise RuntimeError("Comparison failed: at least one file is empty.")


ta = ta[:n]
Va = Va[:n]
Vn = Vn[:n]
Vb = Vb[:n]


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def max_abs(x, y):
    return np.max(np.abs(x - y))


rmse_analytic_numeric = rmse(Va, Vn)
rmse_analytic_builtin = rmse(Va, Vb)
rmse_numeric_builtin  = rmse(Vn, Vb)

max_analytic_numeric = max_abs(Va, Vn)
max_analytic_builtin = max_abs(Va, Vb)
max_numeric_builtin  = max_abs(Vn, Vb)


print("Compared points =", n)
print()
print("RMSE analytic vs numeric =", rmse_analytic_numeric)
print("MAX_ABS analytic vs numeric =", max_analytic_numeric)
print()
print("RMSE analytic vs builtin =", rmse_analytic_builtin)
print("MAX_ABS analytic vs builtin =", max_analytic_builtin)
print()
print("RMSE numeric vs builtin =", rmse_numeric_builtin)
print("MAX_ABS numeric vs builtin =", max_numeric_builtin)


plt.figure(figsize=(10, 6))

plt.plot(ta, Va, label="NESTML analytic")
plt.plot(ta, Vn, "--", label="NESTML numeric odeint/thrust")
plt.plot(ta, Vb, ":", label="built-in iaf_psc_exp")

plt.xlabel("time [ms]")
plt.ylabel("V_m absolute [mV]")
plt.title("IAF PSC EXP comparison: analytic vs numeric vs built-in")
plt.legend()
plt.tight_layout()
plt.savefig(plot_file, dpi=200)
plt.close()

print()
print("Saved plot:", plot_file)
