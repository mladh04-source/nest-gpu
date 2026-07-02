#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=brunel_timing_sweep_%j.out
#SBATCH --error=brunel_timing_sweep_%j.err

ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

export LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):${LD_LIBRARY_PATH}"

python3 /p/project1/cslns/natouf1/test_brunel_timing_sweep.py \
    --neurons 1000 2000 5000 \
    --families iaf aeif \
    --sim-time 1000.0 \
    --compare-script /p/project1/cslns/natouf1/test_brunel_compare_overlay.py \
    --outdir /p/project1/cslns/natouf1/brunel_timing_out_${SLURM_JOB_ID}
