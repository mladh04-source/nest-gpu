#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=brunel_compare_overlay_%j.out
#SBATCH --error=brunel_compare_overlay_%j.err

ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

export LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):${LD_LIBRARY_PATH}"

N_NEURONS=1000

python3 /p/project1/cslns/natouf1/test_brunel_compare_overlay.py \
    --family all \
    --n-neurons ${N_NEURONS} \
    --outdir /p/project1/cslns/natouf1/brunel_compare_out_${SLURM_JOB_ID}
