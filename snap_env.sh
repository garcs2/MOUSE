#!/bin/bash

#SBATCH --partition=short
#SBATCH --time=0-04:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --wckey=edu_res
#SBATCH --error=mouse.err.%J
#SBATCH --output=mouse.txt.%J

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /home/garcsamu/OpenMC/MOUSE/

# Activate conda environment
source ~/miniforge/etc/profile.d/conda.sh
conda activate mouse

# Set environment
export OPENMC_CROSS_SECTIONS=/hpc-common/data/openmc/endfb-viii.0-hdf5/cross_sections.xml
export PYTHONPATH="${PYTHONPATH}:/home/garcsamu/OpenMC/MOUSE"

# Set TMPDIR to shared directory
export TMPDIR="${HOME}/watts_runs/${SLURM_JOB_ID}/tmp"
mkdir -p $TMPDIR

# Run with srun to launch MPI
cd examples/
python watts_exec_LTMR.py ${SLURM_NNODES}