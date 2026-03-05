import numpy as np
import glob

analytic_file = sorted(glob.glob("output_vm_analytic_*.txt"))[-1]
numeric_file  = sorted(glob.glob("output_vm_numeric_*.txt"))[-1]

print("Using:", analytic_file)
print("Using:", numeric_file)

a = np.loadtxt(analytic_file)
n = np.loadtxt(numeric_file)

min_len = min(len(a), len(n))
a = a[:min_len]
n = n[:min_len]

rmse = np.sqrt(np.mean((a[:,1] - n[:,1])**2))

print("Samples compared:", min_len)
print("RMSE:", rmse)
print("Relative RMSE:", rmse / np.mean(np.abs(a[:,1])))
