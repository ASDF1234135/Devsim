"""§2.9 real-vs-real discriminator floor: how well can real messages from two DIFFERENT developers be told apart?
Surface baseline (char n-gram TF-IDF + logistic regression, §3.5 candidate C). Train/test split by session,
classes balanced. Uses train+dev developers only.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from anonymize import anon

D = Path(__file__).resolve().parents[2] / "data"
rng = np.random.default_rng(0)
cp = pd.read_parquet(D / "processed/checkpoints.parquet", columns=["session_id", "user_id", "split", "real_message"])
cp = cp[cp.split.isin(["train", "dev"])].dropna(subset=["real_message", "user_id"])
ok = cp.groupby("user_id").agg(n=("real_message", "size"), s=("session_id", "nunique"))
users = ok[(ok.n >= 200) & (ok.s >= 10)].index.to_numpy()
by = {u: g for u, g in cp[cp.user_id.isin(users)].groupby("user_id")}

res = []
while len(res) < 50:
    a, b = rng.choice(users, 2, replace=False)
    n = min(len(by[a]), len(by[b]), 1000)
    d = pd.concat([by[a].sample(n, random_state=0), by[b].sample(n, random_state=0)])
    y = np.r_[np.zeros(n), np.ones(n)]
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(d, y, d.session_id))
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
    text = d.real_message.map(anon).str[:2000].to_numpy()
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50000)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(v.fit_transform(text[tr]), y[tr])
    p = clf.predict_proba(v.transform(text[te]))[:, 1]
    res.append((accuracy_score(y[te], p > .5), roc_auc_score(y[te], p)))

acc, auc = np.array(res).T
out = {"pairs": len(res), "eligible_developers": len(users), "accuracy_mean": float(acc.mean()),
       "accuracy_p10_p90": [float(np.percentile(acc, 10)), float(np.percentile(acc, 90))], "auc_mean": float(auc.mean())}
(D / "splits/disc_floor.json").write_text(json.dumps(out, indent=1))
print(out)
