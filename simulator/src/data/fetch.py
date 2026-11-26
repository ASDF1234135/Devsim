"""Phase 1 §2.1: pin + download raw data. Run: python simulator/src/data/fetch.py
SWE-chat is gated: accept terms at hf.co/datasets/SALT-NLP/SWE-chat, then set HF_TOKEN."""
import json, os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
# frozen data version for Phase 1; the datasets keep growing, so never default to latest
REVISIONS = {"yfwu/SWE-Together": os.environ.get("SWE_TOGETHER_REV", "2c0058153262a333ed1a203a8352d070361e1b30"),
             "SALT-NLP/SWE-chat": os.environ.get("SWE_CHAT_REV", "202b071f18e03c79df5a7287565ce2ebc5ee7756")}

def fetch(repo, files):
    rev = REVISIONS[repo]
    for f in files:
        hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=RAW / {"yfwu/SWE-Together": "swe_together"}.get(repo, repo.split("/")[1]))
    return rev

if __name__ == "__main__":
    m = {"swe_together": fetch("yfwu/SWE-Together", ["swe_together.jsonl", "data/test-00000-of-00001.parquet"])}
    tables = ["sessions", "conversations", "session_logs", "repositories"]
    try:
        m["swe_chat"] = fetch("SALT-NLP/SWE-chat", [f"{t}.parquet" for t in tables])
    except Exception as e:
        sys.exit(f"SWE-chat download failed ({type(e).__name__}): need HF login + accepted access terms. SWE-Together done.")
    (RAW / "manifest.json").write_text(json.dumps(m, indent=2))
