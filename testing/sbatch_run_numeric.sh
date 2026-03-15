#!/bin/bash
#SBATCH --account=slns
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=numeric_%j.out
#SBATCH --error=numeric_%j.err

# Load modules
ml Stages/2026 GCC OpenMPI CUDA GSL Python SciPy-Stack mpi4py CMake Autotools

# Path to NUMERIC build
source /p/project1/cslns/natouf1/nest-gpu/install_numeric/bin/nestgpu_vars.sh

export LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):${LD_LIBRARY_PATH}"


export OUTFILE=output_vm_numeric.txt
export PLOT_FILE=iaf_psc_exp_neuron_nestml_plot_numeric.png
python3 /p/project1/cslns/natouf1/test_iaf_psc_exp_neuron_nestml_compare.py

