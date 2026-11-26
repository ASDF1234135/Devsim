#!/bin/bash
# CPU job: download pinned data from Hugging Face and rebuild everything that needs no model.
# Submit from the repo root:  sbatch simulator/hpc/stage1_data.sh
#SBATCH --job-name=sim-stage1-data
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=stage1-data-%j.log
# Add your cluster's --partition / --account lines here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .venv/bin/activate ] && source .venv/bin/activate
: "${HF_TOKEN:?export HF_TOKEN (account must have accepted the SWE-chat access terms)}"
S=simulator/src/data

python $S/fetch.py            # ~6.5 GB; if compute nodes have no internet, run this line on the login node
python $S/slim.py
python $S/build_turns.py
python $S/map_swt.py
python $S/splits_baselines.py
python $S/example_bank.py
python $S/swt_gold.py
python $S/disc_floor.py
echo "stage1 data steps done"
