"""Upload Stage 1 results to a PRIVATE Hugging Face dataset repo (needs a write-scoped HF_TOKEN).
  python simulator/src/data/push_results.py <user-or-org>/<repo>
The files derive from gated SWE-chat data (user-written content), so the repo must stay private.
Not uploaded: raw data, conv_slim and checkpoints.parquet (un-anonymized; rebuild them instead).
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi

D = Path(__file__).resolve().parents[2] / "data"
ALLOW = ["splits/*", "processed/example_bank_train.parquet", "processed/length_stats_train.json",
         "processed/goal_cards_*.jsonl", "processed/swt_gold.jsonl", "processed/swt_mapping.parquet"]

if len(sys.argv) != 2:
    sys.exit(__doc__)
repo = sys.argv[1]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
if not api.dataset_info(repo).private:  # exist_ok does not change an existing public repo
    sys.exit(f"{repo} is PUBLIC; refusing to upload.")
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=D, allow_patterns=ALLOW,
                  commit_message="stage 1 results")
print("uploaded to private dataset", repo)
