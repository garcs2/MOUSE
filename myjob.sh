#!/bin/bash

# Submit this script with: sbatch thefilename

#SBATCH --time=4:00:00 # walltime
#SBATCH --ntasks-per-node=48 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1 # number of nodes
#SBATCH --wckey edu_class # Project Code
#SBATCH -J "exmple" # job name
#SBATCH --mail-user=botros.hanna@inl.gov # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# MOUSE Directory
cd /home/username/projects/MOUSE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS (Pyton Environment where OpenMC and Watts are installed)
source activate /home/username/mouse_env

# Your job commands go here
python -m examples.watts_GCMR_Design_reflector > output_ref_GCMR.txt