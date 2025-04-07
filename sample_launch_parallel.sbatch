#!/bin/bash
#SBATCH --job-name=lm-eval-harness
#SBATCH --account=< PROJECT NO >
#SBATCH --time=00:05:00
#SBATCH --partition=dev-g
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --error=slurm_out/slurm_%j.err

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
export HF_HOME=< HF_HOME >
export HF_TOKEN=< TOKEN >	# not required
export PYTHONUSERBASE=< PYTHONUSERBASE >

echo "JOB $SLURM_JOBID START TIME: $(date)"

echo "JOBNAME: $SLURM_JOB_NAME"

MODEL_NAME="< ACCOUNT/MODEL_NAME >"
DTYPE="auto"
NUM_FEWSHOT=3

CMD_PARAMS=(
  "--model" "hf"
  "--model_args" "pretrained=$MODEL_NAME,tokenizer=$MODEL_NAME,dtype=$DTYPE,parallelize=True,max_memory_per_gpu=60GB"
  "--tasks" "finbench_multiple_choice"
  "--num_fewshot" "$NUM_FEWSHOT"
  "--output_path" "./output/"
  "--trust_remote_code"
  "--max_batch_size" "20"
  "--write_out"
  "--log_samples"
  "--seed" "42"
)

CMD="python -m lm_eval ${CMD_PARAMS[@]}"

srun $CMD 

echo "JOB $SLURM_JOBID END TIME: $(date)"
