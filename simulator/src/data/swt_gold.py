"""§2.8 / §4.2: Axis A gold for the mapped tasks, plus the frozen live-dev / live-test split (by task).
SWE-Together already ships atomic intents: intent_kind is the intent type, source_turn the position.
shortcut: no trigger-condition field and no human spot check yet; both needed before Axis A claims.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parents[2] / "data"
t = pd.read_parquet(D / "raw/swe_together/data/test-00000-of-00001.parquet").set_index("task_id")
m = pd.read_parquet(D / "processed/swt_mapping.parquet").dropna(subset=["session_id"]).sort_values("task_id")
order = np.random.default_rng(0).permutation(len(m))
live = np.where(order < len(m) // 2, "live-dev", "live-test")
with open(D / "processed/swt_gold.jsonl", "w", encoding="utf8") as f:
    for r, split in zip(m.itertuples(), live):
        f.write(json.dumps({"task_id": r.task_id, "session_id": r.session_id, "match_score": r.match_score,
                            "live_split": split, "intents": json.loads(t.loc[r.task_id].oracle_intents)},
                           ensure_ascii=False) + "\n")
pd.DataFrame({"task_id": m.task_id, "live_split": live}).to_csv(D / "splits/live_split.csv", index=False)
print(len(m), "tasks;", pd.Series(live).value_counts().to_dict())
