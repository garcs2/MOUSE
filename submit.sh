#!/bin/bash

#SBATCH --job-name=3D_Study
#SBATCH --partition=short
#SBATCH --time=5:58:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --wckey=edu_res
#SBATCH --error=3D.err.%J
#SBATCH --output=3D.txt.%J

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /home/garcsamu/OpenMC/MOUSE/

source ~/miniforge/etc/profile.d/conda.sh
conda activate snap

export OPENMC_CROSS_SECTIONS=/hpc-common/data/openmc/endfb-viii.0-hdf5/cross_sections.xml
export PYTHONPATH="${PYTHONPATH}:/home/garcsamu/OpenMC/MOUSE"



export HDF5_USE_FILE_LOCKING=FALSE

cd examples
python watts_exec_LTMR_3D.py