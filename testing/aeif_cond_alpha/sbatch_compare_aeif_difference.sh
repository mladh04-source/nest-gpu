#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=compare_aeif_difference_%j.out
#SBATCH --error=compare_aeif_difference_%j.err

ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

export LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):${LD_LIBRARY_PATH}"

python3 /p/project1/cslns/natouf1/compare_aeif_difference.py
