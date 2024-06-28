
source _GITIGNORE/secrets.sh

# NOTE adapt these to your HPC
export SBATCH_ARGS="--cpus-per-task=8 --mem=40G --partition=gpu --gres=gpu:a100-80g:1 $SECRET_SBATCH_ARGS"	

if [ -z "$GITHUB_TOKEN" ]; then
  export GITHUB_TOKEN="GITHUB_TOKEN-UNDEFINED"
fi
