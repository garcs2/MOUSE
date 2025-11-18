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

# Load ONLY the OpenMPI module (not py-openmc)
module load openmpi/5.0.5-gcc-13.3.0-lx62

# Set MPI environment variables
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=vader,self,tcp
export OMPI_MCA_btl_base_warn_component_unused=0
export OMPI_MCA_orte_tmpdir_base=/tmp
export PMIX_MCA_gds=hash

# Add OpenMC executable to PATH
OPENMC_BIN_PATH="/apps/spack/opt/gcc-13.3.0/openmc-0.15.0-t3h5rpzqw4f3amku3p6ahvb2gcwl3atb/bin"
export PATH="${OPENMC_BIN_PATH}:${PATH}"

# Set environment
export OPENMC_CROSS_SECTIONS=/hpc-common/data/openmc/endfb-viii.0-hdf5/cross_sections.xml
export PYTHONPATH="${PYTHONPATH}:/home/garcsamu/OpenMC/MOUSE"

# Set TMPDIR to shared directory
export TMPDIR="${HOME}/watts_runs/${SLURM_JOB_ID}/tmp"
mkdir -p $TMPDIR

# Verify
echo "Python: $(which python)"
echo "OpenMC: $(which openmc)"
echo "NumPy from: $(python -c 'import numpy; print(numpy.__file__)')"

# Run with srun to launch MPI
cd examples/
python watts_exec_LTMR.py 