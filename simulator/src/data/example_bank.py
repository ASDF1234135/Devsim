"""§2.5: example bank from TRAIN only (anonymized, deduped) + L2 length statistics + persona labels."""
import json
from pathlib import Path
import pandas as pd
from anonymize import anon

D = Path(__file__).resolve().parents[2] / "data"

cp = pd.read_parquet(D / "processed/checkpoints.parquet")
b = cp[cp.split == "train"].copy()
persona = pd.read_parquet(D / "raw/SWE-chat/sessions.parquet", columns=["session_id", "user_persona"])
b = b.merge(persona, on="session_id", how="left")
for c in ("real_message", "agent_last_response", "agent_thinking_snippet", "prefix_user", "first_prompt", "files_touched"):
    b[c] = b[c].map(anon, na_action="ignore")

# length stats (L2 option 1) use all train messages, before dedup removes frequent short replies
m = b[b.real_action == "message"].assign(chars=lambda x: x.real_message.str.len())  # content is capped at 8000 chars
q = m.groupby(["behaviour", "agent_turn_type"]).chars.quantile([.1, .25, .5, .75, .9]).unstack()
stats = [{"behaviour": k[0], "agent_turn_type": k[1], "n": int(((m.behaviour == k[0]) & (m.agent_turn_type == k[1])).sum()),
          "chars_q": {str(c): float(v) for c, v in row.items()}} for k, row in q.iterrows()]
(D / "processed/length_stats_train.json").write_text(json.dumps(stats, indent=1))

n0 = len(b)
quiet = b[b.real_action == "no_op"]
quiet = quiet.sample(min(len(quiet), 20000), random_state=0)
b = pd.concat([quiet, b[b.real_action == "message"].drop_duplicates("real_message")])
b.drop(columns=["user_id", "repo_id", "comp"]).to_parquet(D / "processed/example_bank_train.parquet", index=False)
assert set(b.split) == {"train"}
print(n0, "->", len(b), "rows;", b.behaviour.value_counts().to_dict())
