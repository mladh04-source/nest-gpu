#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=brunel_iaf_builtin_%j.out
#SBATCH --error=brunel_iaf_builtin_%j.err

ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

source /p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh

export LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):${LD_LIBRARY_PATH}"

N_NEURONS=1000

python3 /p/project1/cslns/natouf1/test_brunel_iaf_psc_exp_builtin.py ${N_NEURONS}
