"""§2.7 splits, §2.9 baselines + real-vs-real floor, §2.10 time model v0, §2.6 type inventory.
Unit of split = connected component of (developer, repo), so no developer or repo spans splits.
Null user_id (31% of sessions) is not a link: those sessions group by repo only.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data"
rng = np.random.default_rng(0)

s = pd.read_parquet(D / "raw/SWE-chat/sessions.parquet", columns=["session_id", "repo_id", "user_id", "agent"])
cp = pd.read_parquet(D / "processed/checkpoints.parquet")
cp = cp.drop(columns=["split", "comp", "agent", "user_id", "repo_id"], errors="ignore")  # safe to rerun
(D / "splits").mkdir(exist_ok=True)
swt = pd.read_parquet(D / "processed/swt_mapping.parquet")

par = {}
def find(x):
    while par.setdefault(x, x) != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for r, u in zip(s.repo_id, s.user_id):
    if isinstance(u, str): par[find("r:" + r)] = find("u:" + u)
s["comp"] = [find("r:" + r) for r in s.repo_id]

n_cp = cp.groupby("session_id").size()
s["n_cp"] = s.session_id.map(n_cp).fillna(0).astype(int)
s["split"] = None
s.loc[s.session_id.isin(swt.session_id.dropna()), "split"] = "swt_overlap"  # excluded from train/dev/test (§2.7)

# greedy balanced assignment of whole components; mapped sessions don't count toward targets
w = s[s.split.isna()].groupby("comp").n_cp.sum().sample(frac=1, random_state=0).sort_values(ascending=False, kind="stable")
tgt = {"train": .70, "dev": .15, "test": .15}; got = dict.fromkeys(tgt, 0); tot = w.sum(); assign = {}
for c, n in w.items():
    k = max(tgt, key=lambda k: tgt[k] - got[k] / tot); assign[c] = k; got[k] += n
s.loc[s.split.isna(), "split"] = s.comp.map(assign)
s[["session_id", "repo_id", "user_id", "agent", "comp", "split"]].to_parquet(D / "splits/splits.parquet", index=False)
cp = cp.merge(s[["session_id", "split", "comp", "agent", "user_id", "repo_id"]], on="session_id")
cp.to_parquet(D / "processed/checkpoints.parquet", index=False)

# leakage check
chk = s[s.split != "swt_overlap"]
for key in ("repo_id", "user_id"):
    k = chk.dropna(subset=[key]).groupby(key).split.nunique()
    assert (k == 1).all(), f"{key} spans splits"

# ---- baselines on dev: no_op log loss + the v1 behaviour distribution ----
cp["act"] = cp.behaviour
tr, dv, te = (cp[cp.split == k] for k in ("train", "dev", "test"))
ll = lambda p, y: float(-np.mean(np.log(np.clip(np.where(y, p, 1 - p), 1e-6, 1))))
y = (dv.real_action == "no_op").to_numpy()
p_type = dv.agent_turn_type.map(tr.groupby("agent_turn_type").real_action.apply(lambda x: (x == "no_op").mean())).fillna((tr.real_action == "no_op").mean())
base = {
    "always_no_op": {"acc_noop": float(y.mean()), "logloss": ll(np.full(len(y), 1 - 1e-6), y)},
    "empirical_by_turn_type": {"logloss": ll(p_type.to_numpy(), y)},
    "global_rate": {"logloss": ll(np.full(len(y), (tr.real_action == "no_op").mean()), y)},
}

LABELS = sorted(tr.behaviour.unique())
def multi_ll(table, rows):  # table: DataFrame turn_type x label of train probabilities
    p = table.reindex(index=rows.agent_turn_type, columns=LABELS).to_numpy()
    return float(-np.mean(np.log(np.clip(p[np.arange(len(rows)), [LABELS.index(b) if b in LABELS else 0 for b in rows.behaviour]], 1e-6, 1))))
by_type = pd.crosstab(tr.agent_turn_type, tr.behaviour, normalize="index")
glob = tr.behaviour.value_counts(normalize=True).reindex(LABELS)
flat = pd.DataFrame([glob] * len(by_type), index=by_type.index)
base["behaviour_labels"] = LABELS
base["behaviour_logloss"] = {"global_rate": multi_ll(flat, dv), "empirical_by_turn_type": multi_ll(by_type, dv)}

# ---- leave-one-agent-out folds (§2.7): train without the agent, evaluate on its dev rows ----
loao = {a: {"train_checkpoints": int((tr.agent != a).sum()), "heldout_dev_checkpoints": int((dv.agent == a).sum())}
        for a in dv.agent.value_counts().index if (dv.agent == a).sum() >= 500}

# ---- real-vs-real floor: split train+dev components in half, JS within turn type, fixed weights ----
def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1)), 0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def gap(a, b, weights):
    out = 0
    for t, wt in weights.items():
        pa = a[a.agent_turn_type == t].act.value_counts(normalize=True)
        pb = b[b.agent_turn_type == t].act.value_counts(normalize=True)
        idx = pa.index.union(pb.index)
        out += wt * js(pa.reindex(idx, fill_value=0).to_numpy(), pb.reindex(idx, fill_value=0).to_numpy())
    return out

pool = cp[cp.split.isin(["train", "dev"])]
weights = pool.agent_turn_type.value_counts(normalize=True).to_dict()
comps = pool.comp.unique(); gaps = []
for _ in range(100):
    half = set(rng.choice(comps, len(comps) // 2, replace=False))
    m = pool.comp.isin(half)
    gaps.append(gap(pool[m], pool[~m], weights))
floor = {"js_mean": float(np.mean(gaps)), "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]}

# ---- time model v0: trimmed median user latency (idle time included, so trim top 10%) ----
lat = cp.user_latency_s.dropna(); lat = lat[(lat >= 0) & (lat <= lat.quantile(0.9))]
time_model = {"user_latency_median_s": float(lat.median()),
              "by_turn_type": cp[cp.user_latency_s.between(0, lat.max())].groupby("agent_turn_type").user_latency_s.median().to_dict(),
              "agent_turn_median_s": float(cp.agent_turn_s.dropna().median())}

report = {
    "n_sessions": int(len(s)), "n_checkpoints": int(len(cp)),
    "split_sessions": s.split.value_counts().to_dict(), "split_checkpoints": cp.split.value_counts().to_dict(),
    "no_op_rate_by_split": cp.groupby("split").apply(lambda g: float((g.real_action == "no_op").mean())).to_dict(),
    "turn_type_by_split": pd.crosstab(cp.agent_turn_type, cp.split).to_dict(),
    "leave_one_agent_out_folds": loao,
    "behaviour_share_by_split": pd.crosstab(cp.split, cp.behaviour, normalize="index").round(4).to_dict("index"),
    "sessions_per_agent": s.agent.value_counts().to_dict(),
    "baselines_dev": base, "real_vs_real_floor_js": floor, "time_model_v0": time_model,
    "no_calibration_data_types": [t for t, n in dv.agent_turn_type.value_counts().items() if n < 200],
}
(D / "splits/stage1_report.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({k: report[k] for k in ("split_sessions", "split_checkpoints", "no_op_rate_by_split", "baselines_dev", "real_vs_real_floor_js", "leave_one_agent_out_folds", "behaviour_share_by_split")}, indent=1, default=str))
