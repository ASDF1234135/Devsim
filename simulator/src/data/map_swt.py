"""§2.8: map SWE-Together tasks to SWE-chat sessions.
Repo alone is too coarse (entireio/cli has ~2.8k sessions), so score candidates by how many of the task's
oracle-intent verbatim excerpts appear in the session's user prompts; keep the best session if >=50% match.
Needs checkpoints.parquet (real_message / first_prompt) from build_turns.py.
"""
import json, re
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[2] / "data"
norm = lambda s: re.sub(r"\s+", " ", (s or "").lower()).strip()

t = pd.read_parquet(D / "raw/swe_together/data/test-00000-of-00001.parquet")
s = pd.read_parquet(D / "raw/SWE-chat/sessions.parquet", columns=["session_id", "repo_id", "user_id"])
cp = pd.read_parquet(D / "processed/checkpoints.parquet", columns=["session_id", "real_message", "first_prompt"])
msgs = cp.groupby("session_id").real_message.apply(lambda x: norm(" || ".join(m for m in x if m))).to_dict()
first = cp.groupby("session_id").first_prompt.first().map(norm).to_dict()
s["repo_l"] = s.repo_id.str.lower()

rows = []
for r in t.itertuples():
    ex = [norm(i["verbatim_excerpt"]) for i in json.loads(r.oracle_intents) if i.get("verbatim_excerpt")]
    cand = s[s.repo_l == (r.repo or "").lower()].session_id
    best = (0.0, None)
    for sid in cand:
        text = msgs.get(sid, "")
        score = sum(e in text for e in ex) / len(ex) if ex else 0.0
        if score > best[0]: best = (score, sid)
    rows.append({"task_id": r.task_id, "repo": r.repo, "n_excerpts": len(ex), "n_candidates": len(cand),
                 "match_score": best[0], "session_id": best[1] if best[0] >= 0.5 else None})
m = pd.DataFrame(rows)
m.to_parquet(D / "processed/swt_mapping.parquet", index=False)
print("mapped:", m.session_id.notna().sum(), "of", len(m), "| candidates exist for", (m.n_candidates > 0).sum())
