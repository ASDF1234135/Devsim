#!/bin/bash
# GPU job: start a local vLLM server, then build goal cards and run the LLM baselines against it.
# Needs stage1_data.sh finished. Submit from the repo root:
#   sbatch simulator/hpc/stage1_models.sh                 # pilot: 50 cards per split
#   CARD_LIMIT=0 sbatch simulator/hpc/stage1_models.sh    # all train + dev sessions (resumable)
#SBATCH --job-name=sim-stage1-models
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=stage1-models-%j.log
# Add your cluster's --partition / --account lines here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -a; source .env; set +a      # GOAL_CARD_* settings

PORT="${VLLM_PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"
export GOAL_CARD_BASE_URL="http://127.0.0.1:$PORT/v1"     # overrides .env
export GOAL_CARD_API_KEY="local-${SLURM_JOB_ID:-$$}-$RANDOM$RANDOM"  # the node may be shared
: "${GOAL_CARD_MODEL:?set GOAL_CARD_MODEL in .env to a Hugging Face model id}"

VLLM_API_KEY="$GOAL_CARD_API_KEY" vllm serve "$GOAL_CARD_MODEL" --host 127.0.0.1 --port "$PORT" \
  --max-model-len "${VLLM_MAX_LEN:-16384}" ${VLLM_EXTRA_ARGS:-} > "vllm-${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do      # up to 30 min for first-time weight download + load
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see vllm log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM not ready"; exit 1; }

S=simulator/src/data
for split in dev train; do     # test cards are built only after freezing (--final)
  python $S/goal_cards.py --split $split --limit "${CARD_LIMIT:-50}" --workers "${CARD_WORKERS:-16}"
done
python $S/llm_baselines.py --n "${BASELINE_N:-500}"
echo "stage1 model steps done"
