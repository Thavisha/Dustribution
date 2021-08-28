#!/bin/bash -l
# NOTE the -l flag!

# Name of job
#SBATCH -J MW3Ddense

# Path for stdout
#SBATCH -o slurm_stdout/MW3Ddense_%j.output  ### %j-job id

# Path for stderr
#SBATCH -e slurm_stderr/MW3Ddense_%j.error

# Get status mails about your jobs
#SBATCH --mail-user dharmawardena@mpia.de
#SBATCH --mail-type=ALL


#SBATCH --nodes=1 ##Number of compute nodes (i.e. physical machines) that will be used

#SBATCH --ntasks=1 ##Number of instances to start - so when we manually code up main and worker jobs, this is > 1, but if we let python sort out the threading/multiprocessing by itself, this is always 1

##SBATCH --partition=four-wks ##Not needed for Raven

# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

##SBATCH --nodelist=karun-node07 ##Not needed for Raven
#SBATCH -D /ptmp/thadhar/Orion_20Grid_CPU

#SBATCH --cpus-per-task=144 ##12 24
#SBATCH --get-user-env
##SBATCH --exclusive=user  ##Causes problem for now by blocking the node to a single user
# The following is *essential* if you want more than one task per node, i.e. limit the memory allocation
##SBATCH --mem=2048000  ##Amount of memory needed per node #THIS IS HOW YOU CHOOSE THE BIG MEMORY MACHINES


##module load anaconda/3/2020.02
##source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2020.02/etc/profile.d/conda.sh
##source activate /u/thadhar/conda-envs/MWExtEnv


##srun /u/thadhar/conda-envs/MWExtEnv/bin/python run_Gpytorch_Ext_andDens_Pred.py

module purge
module load anaconda/3/2021.05
module load cuda/11.2
module load pytorch/gpu-cuda-11.2/1.9.0
module load gpytorch/gpu-cuda-11.2/pytorch-1.9.0/.1.5.0 ##please note the dot before the gpytorch version number (1.5.0)

PYTHONPATH=/u/thadhar/.local/lib/python3.8/site-packages/:$PYTHONPATH
echo PYTHONPATH = $PYTHONPATH

srun python run_Gpytorch_Ext_andDens_Pred.py




