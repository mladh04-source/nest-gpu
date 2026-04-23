#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=brunel_aeif_builtin_%j.out
#SBATCH --error=brunel_aeif_builtin_%j.err

# Load modules
ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

# Built-in NEST GPU (no odeint build!)
source /p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh

# Make Python shared library visible
export LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):${LD_LIBRARY_PATH}"

# Network size 
N_NEURONS=12500

python3 /p/project1/cslns/natouf1/test_brunel_aeif_cond_alpha_builtin.py ${N_NEURONS}
