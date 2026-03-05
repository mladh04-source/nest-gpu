#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=analytic_%j.out
#SBATCH --error=analytic_%j.err

ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

source /p/project1/cslns/natouf1/nest-gpu/install_analytic/bin/nestgpu_vars.sh

export OUTFILE="$SLURM_SUBMIT_DIR/output_vm_analytic_${SLURM_JOB_ID}.txt"

srun --export=ALL python3 $SLURM_SUBMIT_DIR/test_iaf_psc_exp_compare.py

