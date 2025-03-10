#!/bin/bash
#SBATCH --job-name=lm-eval-harness
#SBATCH --account=project_000000000
#SBATCH --time=02:30:00
#SBATCH --partition=small-g
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --error=slurm_out/slurm_%j.err

### Sample slurm launch script for lm-evaluation-harness for use in the LUMI HPC-environment ###

# Create log directory if it doesn't exist
mkdir -p slurm_out
# Remove previously created symlinks just in case
rm -f slurm_out/latest.out slurm_out/latest.err
# Create latest log symlinks for current job
ln -s slurm_$SLURM_JOB_ID.out slurm_out/latest.out
ln -s slurm_$SLURM_JOB_ID.err slurm_out/latest.err

# Auto-fail on sbatch errors
set -eox pipefail

# Logging script's variables/commands for future debug needs
set -x

# Sanity check
echo "PWD: $PWD"

module use /appl/local/csc/modulefiles
module load pytorch
export HF_HOME=/scratch/project_000000000/hf_cache
# lm-evaluation-harness needs SacreBLEU, which I installed with `pip install --user` like this:
# https://docs.csc.fi/support/tutorials/python-usage-guide/#__tabbed_1_2
export PYTHONUSERBASE=/projappl/project_000000000/username/lm-venv

echo "JOB $SLURM_JOBID START TIME: $(date)"

echo "JOBNAME: $SLURM_JOB_NAME"

MODEL_NAME="LumiOpen/Poro-34B"
DTYPE="bfloat16"

CMD="python -m lm_eval \
  --model hf \
	--model_args pretrained=$MODEL_NAME,dtype=$DTYPE,parallelize=True \
	--tasks task,names,separated,like,this \
	--output_path output/ \
	--trust_remote_code \
	--batch_size 8"

srun $CMD

echo "JOB $SLURM_JOBID END TIME: $(date)"