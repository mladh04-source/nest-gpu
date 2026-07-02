#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=profile_aeif_old_nestml_%j.out
#SBATCH --error=profile_aeif_old_nestml_%j.err

module --force purge

ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

source /p/project1/cslns/natouf1/nest-gpu_old_nestml/install_old_nestml/bin/nestgpu_vars.sh

export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH}"
export MPLBACKEND=Agg

NSYS="/p/software/jusuf/stages/2024/software/Nsight-Systems/2023.3.1-GCCcore-12.3.0/bin/nsys"
RUN_BASE="/p/project1/cslns/natouf1/nsys_profiles/aeif_old_nestml"

echo "Running on node:"
hostname

echo "Loaded modules:"
module list

echo "CUDA devices:"
nvidia-smi

echo "nsys path:"
echo "$NSYS"
"$NSYS" --version

for N in 1000 2000 5000
do
  echo "========================================"
  echo "Profiling AEIF old NESTML with ${N} neurons"
  echo "========================================"

  RUN_DIR="${RUN_BASE}/${N}"
  OUT_BASE="${RUN_DIR}/aeif_old_nestml_${N}_2024_nsys"

  mkdir -p "$RUN_DIR"

  srun "$NSYS" profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --cpuctxsw=none \
    --stats=true \
    --force-overwrite=true \
    --output="$OUT_BASE" \
    python3 -u /p/project1/cslns/natouf1/test_brunel_compare_overlay_old_nestml.py \
      --worker \
      --family aeif \
      --impl old_nestml \
      --n-neurons "$N" \
      --sim-time 1000.0 \
      --outdir "$RUN_DIR"
done
