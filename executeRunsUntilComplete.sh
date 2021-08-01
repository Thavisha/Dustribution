#!/bin/bash
# Submit a chain of batch jobs with dependencies
#
# Number of jobs to submit:
NR_OF_JOBS=100
# Batch job script:
JOB_SCRIPT_START=./MWExt_slurm.sh
JOB_SCRIPT_RESTART=./MWExt_slurm_reStartJob.sh
echo "Submitting job chain of ${NR_OF_JOBS} jobs for batch script ${JOB_SCRIPT_START}:"
JOBID=$(sbatch ${JOB_SCRIPT_START} 2>&1 | awk '{print $(NF)}')
echo "  " ${JOBID}
I=1
while [ ${I} -lt ${NR_OF_JOBS} ]; do
    JOBID=$(sbatch --dependency=afternotok:${JOBID} ${JOB_SCRIPT_RESTART} 2>&1 | awk '{print $(NF)}')
    echo "  " ${JOBID}
    let I=${I}+1
done