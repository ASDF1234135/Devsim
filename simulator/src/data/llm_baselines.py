"""§2.9 LLM baselines on a dev sample: persona-only and prompt-based local chat model.
Uses the same local endpoint as the goal cards (.env). They emit one label, not probabilities,
so report accuracy and the label-distribution gap (JS, nats), next to empirical-frequency sampling.
  python simulator/src/data/llm_baselines.py --n 100
"""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anonymize import anon

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "simulator" / "data"
LABELS = ["no_op", "non_pushback", "correction", "failure_report", "rejection"]
ASK = ("You are a developer working with a coding agent. The agent just finished its turn. What do you do next? "
       "Answer with JSON {\"behaviour\": X}, X one of: no_op (send nothing), non_pushback (send a message without pushing back), "
       "correction (correct the agent), failure_report (report something failed), rejection (reject the work).")


def js(a, b):
    p, q = (np.array([(x == l).sum() for l in LABELS], float) + 0.2 for x in (a, b))
    p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    model = os.environ["GOAL_CARD_MODEL"]
    client = OpenAI(base_url=os.environ["GOAL_CARD_BASE_URL"], api_key=os.environ.get("GOAL_CARD_API_KEY") or "local", timeout=900)
    effort = os.environ.get("GOAL_CARD_REASONING_EFFORT")
    cp = pd.read_parquet(D / "processed/checkpoints.parquet")
    persona = pd.read_parquet(D / "raw/SWE-chat/sessions.parquet", columns=["session_id", "user_persona"]).set_index("session_id").user_persona
    dv = cp[(cp.split == "dev") & (cp.behaviour != "unlabeled")].sample(a.n, random_state=0)
    tr = cp[(cp.split == "train") & (cp.behaviour != "unlabeled")]
    freq = pd.crosstab(tr.agent_turn_type, tr.behaviour, normalize="index").reindex(columns=LABELS, fill_value=0)
    rng = np.random.default_rng(0)
    preds = {"empirical_sample": [rng.choice(LABELS, p=freq.loc[t] / freq.loc[t].sum()) for t in dv.agent_turn_type]}

    def ask(system, user):
        r = client.chat.completions.create(model=model, temperature=0.7, seed=0, max_tokens=30,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}, extra_body={"reasoning_effort": effort} if effort else None)
        try: x = json.loads(r.choices[0].message.content)["behaviour"]
        except (ValueError, KeyError, TypeError): return "parse_failed"
        return x if x in LABELS else "parse_failed"

    for name in ("persona_only", "prompt_based"):
        preds[name] = []
        for i, r in enumerate(dv.itertuples(), 1):
            if name == "persona_only":  # persona is a hindsight session label from the dataset
                system, user = f"{ASK} Your persona: {persona.get(r.session_id) or 'Other'}.", "Choose now."
            else:
                system = ASK
                user = anon(f"Task: {r.first_prompt[:800]}\nYour earlier messages: {r.prefix_user[-1000:]}\n"
                            f"Agent turn type: {r.agent_turn_type}\nAgent's last reply: {r.agent_last_response[:1200]}")
            preds[name].append(ask(system, user))
            if i % 10 == 0: print(name, i, flush=True)

    y = dv.behaviour.to_numpy()
    out = {"n": len(dv), "model": model, "majority_class_acc": float((y == tr.behaviour.mode()[0]).mean())}
    for k, v in preds.items():
        v = np.array(v)
        out[k] = {"accuracy": float((v == y).mean()), "js_to_real": js(y, v), "parse_failed": int((v == "parse_failed").sum())}
    (D / "splits/llm_baselines.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
