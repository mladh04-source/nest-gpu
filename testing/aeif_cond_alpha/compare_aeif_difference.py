import os
import numpy as np
import matplotlib.pyplot as plt

builtin_file = "/p/project1/cslns/natouf1/output_vm_aeif_builtin.txt"
odeint_file = "/p/project1/cslns/natouf1/output_vm_aeif_odeint.txt"
old_nestml_file = "/p/project1/cslns/natouf1/output_vm_aeif_old_nestml.txt"

plot_file_odeint = "/p/project1/cslns/natouf1/aeif_single_neuron_current_odeint_comparison.png"
diff_file_odeint = "/p/project1/cslns/natouf1/difference_vm_aeif_builtin_vs_odeint.txt"

plot_file_old = "/p/project1/cslns/natouf1/aeif_single_neuron_legacy_generated_comparison.png"
diff_file_old = "/p/project1/cslns/natouf1/difference_vm_aeif_builtin_vs_old_nestml.txt"


def compare_and_plot(reference_file,
                     compare_file,
                     compare_label,
                     diff_label,
                     diff_header,
                     plot_title,
                     plot_file,
                     diff_file,
                     compare_color):
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    if not os.path.exists(compare_file):
        raise FileNotFoundError(f"Comparison file not found: {compare_file}")

    ref = np.loadtxt(reference_file)
    cmp = np.loadtxt(compare_file)

    t_ref, V_ref = ref[:, 0], ref[:, 1]
    t_cmp, V_cmp = cmp[:, 0], cmp[:, 1]

    n = min(len(V_ref), len(V_cmp))

    if n == 0:
        raise RuntimeError("Comparison failed: one of the files is empty.")

    t_ref = t_ref[:n]
    t_cmp = t_cmp[:n]
    V_ref = V_ref[:n]
    V_cmp = V_cmp[:n]

    if not np.allclose(t_ref, t_cmp):
        print(f"WARNING: time vectors are not exactly identical for {compare_label}.")
        print("max time difference =", np.max(np.abs(t_ref - t_cmp)))

    diff = V_cmp - V_ref

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    max_abs = np.max(np.abs(diff))

    np.savetxt(
        diff_file,
        np.column_stack((t_ref, diff)),
        header=diff_header
    )

    print()
    print(plot_title)
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

    axes[0].plot(t_ref, V_ref, label="Native AEIF", color="tab:blue")
    axes[0].plot(
        t_ref,
        V_cmp,
        "--",
        label=compare_label,
        color=compare_color,
    )
    axes[0].set_ylabel("Membrane potential $V_m$ [mV]")
    axes[0].set_title(plot_title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ref, diff, label=diff_label, color=compare_color)
    axes[1].axhline(0.0, linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Voltage difference [mV]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved plot:", plot_file)


# Odeint vs built-in
compare_and_plot(
    reference_file=builtin_file,
    compare_file=odeint_file,
    compare_label="Current Odeint AEIF",
    diff_label="Current Odeint AEIF - Native AEIF",
    diff_header="time_ms Vm_odeint_minus_builtin_mV",
    plot_title="AEIF single-neuron comparison: Current Odeint vs Native",
    plot_file=plot_file_odeint,
    diff_file=diff_file_odeint,
    compare_color="tab:orange",
)

# old_nestml vs built-in
compare_and_plot(
    reference_file=builtin_file,
    compare_file=old_nestml_file,
    compare_label="Legacy generated AEIF",
    diff_label="Legacy generated AEIF - Native AEIF",
    diff_header="time_ms Vm_old_nestml_minus_builtin_mV",
    plot_title="AEIF single-neuron comparison: Legacy generated vs Native",
    plot_file=plot_file_old,
    diff_file=diff_file_old,
    compare_color="tab:green",
)
