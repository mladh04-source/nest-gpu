#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=aeif_rk5_%j.out
#SBATCH --error=aeif_rk5_%j.err

# Load modules
ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

# RK5 build (USE_ODEINT_THRUST=0)
source /p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh

# Make Python shared library visible
export LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):${LD_LIBRARY_PATH}"

# Test configuration
export MODEL_NAME=aeif_cond_alpha_alt_neuron_nestml
export OUTFILE=/p/project1/cslns/natouf1/output_vm_aeif_rk5.txt
export PLOT_FILE=/p/project1/cslns/natouf1/plot_aeif_rk5.png
export REFERENCE_FILE=/p/project1/cslns/natouf1/output_vm_aeif_builtin.txt

python3 /p/project1/cslns/natouf1/test_aeif_cond_alpha_compare.py
