#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=profile_iaf_builtin_%j.out
#SBATCH --error=profile_iaf_builtin_%j.err

ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools
ml Nsight-Systems/2025.5.1

source /p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh

export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH}"

mkdir -p /p/project1/cslns/natouf1/nsys_brunel_reports
mkdir -p /p/project1/cslns/natouf1/profile_iaf_builtin_1000

echo "Running on node:"
hostname

echo "CUDA devices:"
nvidia-smi

echo "nsys path:"
which nsys
nsys --version

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --stats=true \
  --force-overwrite=true \
  -o /p/project1/cslns/natouf1/nsys_brunel_reports/iaf_builtin_1000 \
  python3 /p/project1/cslns/natouf1/test_brunel_compare_overlay.py \
    --worker \
    --family iaf \
    --impl builtin \
    --n-neurons 1000 \
    --sim-time 1000.0 \
    --outdir /p/project1/cslns/natouf1/profile_iaf_builtin_1000
