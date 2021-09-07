#!/bin/bash -l
# NOTE the -l flag!

# Name of job
#SBATCH -J MW3Ddense

# Path for stdout
#SBATCH -o slurm_stdout/MW3Ddense_%j.output  ### %j-job id

# Path for stderr
#SBATCH -e slurm_stderr/MW3Ddense_%j.error

# Get status mails about your jobs
#SBATCH --mail-user ENTER EMAIL
#SBATCH --mail-type=ALL


#SBATCH --nodes=1 ##Number of compute nodes (i.e. physical machines) that will be used

#SBATCH --ntasks=1 ##Number of instances to start - so when we manually code up main and worker jobs, this is > 1, but if we let python sort out the threading/multiprocessing by itself, this is always 1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000

##SBATCH --partition=four-wks

# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

##SBATCH --nodelist=karun-node07
#SBATCH -D /ENTER PATH TO slurm_stderr and slurm_stdout FOLDERS

##SBATCH --cpus-per-task=144 ##12 24
#SBATCH --get-user-env
##SBATCH --exclusive=user  ##Causes problem for now by blocking the node to a single user
# The following is *essential* if you want more than one task per node, i.e. limit the memory allocation
##SBATCH --mem=2048000  ##Amount of memory needed per node #THIS IS HOW YOU CHOOSE THE BIG MEMORY MACHINES


srun python run_Gpytorch_Ext_andDens_Pred.py resume_training


















