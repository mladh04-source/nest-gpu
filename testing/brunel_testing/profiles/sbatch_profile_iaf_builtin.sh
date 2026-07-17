#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=profile_iaf_builtin_%j.out
#SBATCH --error=profile_iaf_builtin_%j.err

module --force purge

ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

source /p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh

export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH}"
export MPLBACKEND=Agg

NSYS="/p/software/jusuf/stages/2024/software/Nsight-Systems/2023.3.1-GCCcore-12.3.0/bin/nsys"
RUN_BASE="/p/project1/cslns/natouf1/nsys_profiles/iaf_builtin"

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
  echo "Profiling IAF builtin with ${N} neurons"
  echo "========================================"

  RUN_DIR="${RUN_BASE}/${N}"
  OUT_BASE="${RUN_DIR}/iaf_builtin_${N}_2024_nsys"

  mkdir -p "$RUN_DIR"

  # NEW: The automatic --stats=true output was removed because its table format truncates long CUDA kernel names.
  srun "$NSYS" profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="$OUT_BASE" \
    python3 -u /p/project1/cslns/natouf1/test_brunel_compare_overlay.py \
      --worker \
      --family iaf \
      --impl builtin \
      --n-neurons "$N" \
      --sim-time 1000.0 \
      --outdir "$RUN_DIR"

  # NEW: Generate each Nsight Systems statistics report separately after profiling has finished.
  for REPORT in \
    nvtx_sum \
    osrt_sum \
    cuda_api_sum \
    cuda_gpu_kern_sum \
    cuda_gpu_mem_time_sum \
    cuda_gpu_mem_size_sum
  do
    echo "========================================"
    echo "${REPORT}: ${N} neurons"
    echo "========================================"

    # NEW: CSV format preserves the complete report fields, including full CUDA kernel names.
    # NEW: tee writes the report both to the Slurm output and to a separate CSV file.
    "$NSYS" stats \
      --quiet \
      --report "$REPORT" \
      --format csv \
      "${OUT_BASE}.nsys-rep" \
      | tee "${RUN_DIR}/iaf_builtin_${N}_${REPORT}.csv"
  done
done
