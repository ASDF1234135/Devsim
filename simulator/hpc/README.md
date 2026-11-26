# Running Stage 1 on a Slurm HPC

Data comes from Hugging Face at pinned revisions (`src/data/fetch.py`). Code travels by git or copy
(`simulator/src`, `simulator/hpc`, `simulator/requirements.txt`, `.env.example`). Never copy `.env` into git.

## Setup (login node, repo root)

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r simulator/requirements.txt
export HF_TOKEN=...                                    # account that accepted the SWE-chat terms
cp .env.example .env
```

- **Storage:** raw data plus outputs take about 8 GB. If home is small, put the data on scratch first:
  `mkdir -p $SCRATCH/sim_data && ln -s $SCRATCH/sim_data simulator/data`.
  Do the same for the model cache with `export HF_HOME=$SCRATCH/hf`.
- **Privacy:** SWE-chat is gated and contains user-written text. Keep the data directory readable only by you
  (`chmod 700`).
- **Slurm:** add your `--partition` / `--account` lines to both job scripts.

## 1. Data steps (CPU)

```bash
sbatch simulator/hpc/stage1_data.sh
```

## 2. Model steps (GPU)

Install vLLM (`pip install vllm`, matching your cluster's CUDA). In `.env` set:

- `GOAL_CARD_MODEL`: the Hugging Face id of the model to serve (an Ollama tag such as `qwen3.5:latest` will not work).
- `GOAL_CARD_MAX_INPUT_CHARS=24000` (the job serves 16k tokens of context).
- `GOAL_CARD_REASONING_EFFORT=` (empty). If the model "thinks" by default, turn that off with the server flag your
  vLLM version supports, via `VLLM_EXTRA_ARGS`; otherwise cards fail to parse.

```bash
sbatch simulator/hpc/stage1_models.sh                   # pilot: 50 cards per split
```

Check the pass and parse-failure counts in the log and read some cards before the full run:

```bash
CARD_LIMIT=0 sbatch simulator/hpc/stage1_models.sh      # all dev + train; rerun to resume
```

Delete `simulator/data/processed/goal_cards_*.jsonl` after changing model or prompt: resume skips existing sessions.

## 3. Results

`simulator/data/splits/` has the reports. To send results through Hugging Face (private dataset repo, needs a
write token):

```bash
python simulator/src/data/push_results.py <user-or-org>/<repo>
```

Not covered here: the SWE-Together reproduction run (needs a container runtime and agent API access) and
d3 zero-shot.
