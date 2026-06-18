import os
import numpy as np
import matplotlib.pyplot as plt


base_dir = os.environ.get("BASE_DIR", "/p/project1/cslns/natouf1")

analytic_file = os.path.join(base_dir, "output_vm_analytic.txt")
numeric_file  = os.path.join(base_dir, "output_vm_numeric.txt")
builtin_file  = os.path.join(base_dir, "output_vm_builtin.txt")

plot_file = os.path.join(base_dir, "comparison_plot_iaf_with_diff.png")
diff_file = os.path.join(base_dir, "difference_vm_iaf.txt")


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


t = tb[:n]
ta = ta[:n]
tn = tn[:n]
tb = tb[:n]

Va = Va[:n]
Vn = Vn[:n]
Vb = Vb[:n]


if not np.allclose(ta, tb):
    print("WARNING: analytic and built-in time vectors are not identical.")
    print("max time difference analytic-builtin =", np.max(np.abs(ta - tb)))

if not np.allclose(tn, tb):
    print("WARNING: numeric and built-in time vectors are not identical.")
    print("max time difference numeric-builtin =", np.max(np.abs(tn - tb)))


diff_analytic_builtin = Va - Vb
diff_numeric_builtin  = Vn - Vb
diff_numeric_analytic = Vn - Va


def rmse(diff):
    return np.sqrt(np.mean(diff ** 2))


def mae(diff):
    return np.mean(np.abs(diff))


def max_abs(diff):
    return np.max(np.abs(diff))


print("IAF single neuron numerical comparison")
print("Compared points =", n)
print()

print("analytic - builtin:")
print("RMSE [mV]    =", rmse(diff_analytic_builtin))
print("MAE [mV]     =", mae(diff_analytic_builtin))
print("MAX_ABS [mV] =", max_abs(diff_analytic_builtin))
print()

print("numeric - builtin:")
print("RMSE [mV]    =", rmse(diff_numeric_builtin))
print("MAE [mV]     =", mae(diff_numeric_builtin))
print("MAX_ABS [mV] =", max_abs(diff_numeric_builtin))
print()

print("numeric - analytic:")
print("RMSE [mV]    =", rmse(diff_numeric_analytic))
print("MAE [mV]     =", mae(diff_numeric_analytic))
print("MAX_ABS [mV] =", max_abs(diff_numeric_analytic))


np.savetxt(
    diff_file,
    np.column_stack(
        (
            t,
            diff_analytic_builtin,
            diff_numeric_builtin,
            diff_numeric_analytic,
        )
    ),
    header="time_ms analytic_minus_builtin_mV numeric_minus_builtin_mV numeric_minus_analytic_mV"
)

print()
print("Saved diff data:", diff_file)


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 1]}
)


axes[0].plot(t, Vb, label="built-in iaf_psc_exp")
axes[0].plot(t, Va, "--", label="NESTML analytic")
axes[0].plot(t, Vn, ":", label="NESTML numeric odeint/thrust")
axes[0].set_ylabel("V_m absolute [mV]")
axes[0].set_title("IAF PSC EXP single neuron comparison")
axes[0].legend()
axes[0].grid(True, alpha=0.3)


axes[1].plot(t, diff_analytic_builtin, label="analytic - built-in")
axes[1].plot(t, diff_numeric_builtin, "--", label="numeric - built-in")
axes[1].axhline(0.0, linestyle=":", linewidth=1.0)
axes[1].set_ylabel("diff [mV]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)


axes[2].plot(t, diff_numeric_analytic, label="numeric - analytic")
axes[2].axhline(0.0, linestyle=":", linewidth=1.0)
axes[2].set_xlabel("time [ms]")
axes[2].set_ylabel("diff [mV]")
axes[2].legend()
axes[2].grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig(plot_file, dpi=200)
plt.close()

print("Saved plot:", plot_file)
