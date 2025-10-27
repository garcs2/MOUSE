#!/bin/bash

# Submit this script with: qsub thefilename

#SBATCH --partition=short                              # default general (options: general, short, hbm)
#SBATCH --time=0-04:00:00                              # run time in days-hh:mm:ss (6 hours max for short, 168 hours max for general)
#SBATCH --nodes=32                                     # number of job nodes (max is 168 nodes on general, 336 nodes on short)
#SBATCH --ntasks-per-node=1                            # mpi ranks per node
#SBATCH --cpus-per-task=112                            # threads per mpi rank
#SBATCH --wckey=edu_res                                # project code
#SBATCH --error=mouse.err.%J                          # job error file
#SBATCH --output=mouse.txt.%J                         # job output file

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Navigate to MOUSE root directory
cd /home/garcsamu/OpenMC/MOUSE/

# Activate conda environment FIRST
source ~/miniforge/etc/profile.d/conda.sh
conda activate mouse

# Save the OpenMC executable path before it gets lost
OPENMC_BIN_PATH="/apps/spack/opt/gcc-13.3.0/openmc-0.15.0-t3h5rpzqw4f3amku3p6ahvb2gcwl3atb/bin"

# Prepend ONLY the OpenMC binary directory to PATH
# Put it at the beginning so it's found first, but keep conda's Python
export PATH="${OPENMC_BIN_PATH}:${PATH}"

# Verify setup
echo "Python: $(which python)"
echo "OpenMC: $(which openmc)"
python -c "import watts; print('WATTS found')"

# Set environment variables
export APPTAINERENV_OPENMC_CROSS_SECTIONS=/hpc-common/data/openmc/endfb-viii.0-hdf5/cross_sections.xml
export PYTHONPATH="${PYTHONPATH}:/home/garcsamu/OpenMC/MOUSE"

# Run from examples directory
cd examples/
python watts_exec_LTMR.py ${SLURM_NNODES} > output.txt