#!/bin/bash
#SBATCH --job-name=lm-eval-harness
#SBATCH --account=<omitted>
#SBATCH --time=01:00:00
#SBATCH --partition=dev-g
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --error=slurm_out/slurm_%j.err

# Create log directory if it doesn't exist
mkdir -p slurm_out
# Remove previously created symlinks just in case
rm -f slurm_out/latest.out slurm_out/latest.err
# Create latest log symlinks for current job
ln -s slurm_$SLURM_JOB_ID.out slurm_out/latest.out
ln -s slurm_$SLURM_JOB_ID.err slurm_out/latest.err

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_path> <output_file_path>"
    exit 1
fi

MODEL_NAME="$1"
OUTPUT_PATH="$2"

# Ensure the input folder exists
# If you want to run a model on HF hub, this check should be skipped
#if [[ ! -d "$MODEL_NAME" ]]; then
#    echo "Error: Input folder '$MODEL_NAME' does not exist."
#    exit 1
#fi

# Extract the directory part of the OUTPUT_PATH
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")

# Ensure the output folder exists, create if it doesn't
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Output folder '$OUTPUT_DIR' does not exist. Creating it..."
    mkdir -p "$OUTPUT_DIR"
fi

# Auto-fail on sbatch errors
set -eox pipefail

# Logging script's variables/commands for future debug needs
set -x

# Sanity check
echo "PWD: $PWD"

module use /appl/local/csc/modulefiles
module load pytorch
export HF_HOME=<omitted>
export HF_TOKEN=<omitted>
export PYTHONUSERBASE=<omitted>

# lm_eval has changed so that --output_path is treated as a directory and the
# file outputs are stored in a directory structure under that point.  we don't
# want that.  so now we output to a temporary directory then move the resulting
# file to $OUTPUT_PATH
RANDOM_DIR=$(mktemp -d)

# Exit if the temp directory wasn't created successfully.
if [ ! -e "$RANDOM_DIR" ]; then
    >&2 echo "Failed to create temp directory"
    exit 1
fi

echo Saving temporary results to $RANDOM_DIR

echo "JOB $SLURM_JOBID START TIME: $(date)"

echo "JOBNAME: $SLURM_JOB_NAME"

###########################################
# THIS IS PROBABLY WHAT YOU WANT TO TWEAK #
###########################################

DTYPE="bfloat16"
NUM_FEWSHOT=3

# Set to "False" if using only one GPU, otherwise "True"
PARALLELIZE=False

CMD_PARAMS=(
  "--model" "hf"
  "--model_args" "pretrained=$MODEL_NAME,tokenizer=$MODEL_NAME,dtype=$DTYPE,parallelize=$PARALLELIZE,max_memory_per_gpu=60GB"
  "--tasks" "list"
  "--num_fewshot" "$NUM_FEWSHOT"
  "--output_path" "$RANDOM_DIR"
  "--trust_remote_code"
  "--max_batch_size" "24"
  "--seed" "42"
)

CMD="python -m lm_eval ${CMD_PARAMS[@]}"

srun $CMD 

echo Moving temporary results from $RANDOM_DIR to $OUTPUT_PATH
find "$RANDOM_DIR" -name "results_*.json" -exec mv {} "$OUTPUT_PATH" \;
rm -rf "$RANDOM_DIR"

echo "JOB $SLURM_JOBID END TIME: $(date)"
