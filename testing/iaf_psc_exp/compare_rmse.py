import numpy as np
import matplotlib.pyplot as plt

analytic_file = "output_vm_analytic.txt"
numeric_file = "output_vm_numeric.txt"

a = np.loadtxt(analytic_file)
n = np.loadtxt(numeric_file)

t = a[:,0]
Va = a[:,1]
Vn = n[:,1]

rmse = np.sqrt(np.mean((Va - Vn)**2))

print("RMSE analytic vs numeric =", rmse)

plt.plot(t, Va, label="analytic")
plt.plot(t, Vn, "--", label="numeric")
plt.legend()
plt.xlabel("time (ms)")
plt.ylabel("V_m")
plt.title(f"RMSE = {rmse}")
plt.savefig("comparison_plot.png")

print("Saved comparison_plot.png")

