"""Pass 2 (§2.3, §2.6): one checkpoint per agent turn.

Segmentation: a segment is everything the agent did after user prompt k (or session start) up to the next
user_prompt. The checkpoint sits at the segment end; real_action = the next user prompt, or no_op when the
session ends. shortcut: no mid-segment checkpoints, so silence only appears at session end; upgrade when
queue-op/idle-gap based turn ends are needed for the silence rate.
"""
import json, re
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[2] / "data"
EDIT = {"Edit", "Write", "MultiEdit", "NotebookEdit", "apply_patch", "str_replace_editor", "write", "edit"}
ERR = re.compile(r"\b(error|traceback|failed|exception|FAIL|not found)\b", re.I)
DONE = re.compile(r"\b(done|complete[d]?|implemented|finished|all set|ready for review)\b", re.I)
QUEUE_OPS = ("user_prompt_enqueued", "user_prompt_delivered", "user_prompt_discarded")
PREFIX_USER_N, PREFIX_CHARS = 6, 1500

# Harness-written rows stored as user_prompt; they are not user behaviour (§2.5 cleaning).
INJECTED = re.compile(r"^\s*(\[Request interrupted by user|<command-(name|message)>|<local-command|<environment_context>|"
                      r"<subagent_notification>|<task-notification>|<turn_aborted>|<codex_internal_context|# AGENTS\.md instructions|"
                      r"Another Claude session se|This session is being continued|>>> |<system-reminder>)")
IMAGE_ONLY = re.compile(r"^(\s*\[Image[^\]]*\])+\s*$")
REVIEW_BOT = "The following is the Codex agent history"  # automated approval-review sessions: no human user


def is_injected(text):
    return not text or not text.strip() or bool(INJECTED.match(text) or IMAGE_ONLY.match(text))


def turn_type(f):  # priority order matters; mechanical per §2.6
    if f["plan_submitted"]: return "submits_plan"
    if f["ends_with_question"]: return "asks_question"
    if f["test_failed"]: return "reports_error"
    if f["claims_completion"]: return "claims_completion"
    if f["n_edits"] >= 5: return "large_edit"
    return "ordinary_progress"


def session_checkpoints(g):
    g = g.sort_values("turn_number")
    qops = g[g.turn_type == "queue_operation"]
    g = g[g.turn_type != "queue_operation"]
    prompts = g[g.turn_type == "user_prompt"].content.fillna("")
    if prompts.str.lstrip().str.startswith(REVIEW_BOT).any(): return []
    g = g.drop(prompts.index[prompts.map(is_injected)])
    if g.empty: return []
    up = g.index[g.turn_type == "user_prompt"].tolist()
    pos = {i: k for k, i in enumerate(g.index)}
    rows, hist, edits_seen = [], [], 0
    bounds = [None] + up + [None]  # segment k: after bounds[k], before bounds[k+1]
    for k in range(len(bounds) - 1):
        lo = pos[bounds[k]] + 1 if bounds[k] is not None else 0
        hi = pos[bounds[k + 1]] if bounds[k + 1] is not None else len(g)
        seg = g.iloc[lo:hi]
        if bounds[k] is None and seg.empty: continue
        asst = seg[seg.turn_type == "assistant_response"]
        last_resp = (asst.content.iloc[-1] or "") if len(asst) else ""
        tools = seg[seg.turn_type == "tool_use"]
        res = seg[seg.turn_type == "tool_result"]
        n_edits = int(tools.tool_name.isin(EDIT).sum())
        edits_seen += n_edits
        think = seg[seg.turn_type == "assistant_thinking"]
        f = {
            "ends_with_question": last_resp.rstrip().endswith("?") or last_resp.rstrip().endswith("？"),
            "wrote_files": n_edits > 0, "n_edits": n_edits, "n_tools": len(tools),
            "plan_submitted": bool((tools.tool_name == "ExitPlanMode").any()),
            "test_failed": bool(len(res) and res.content.fillna("").tail(3).map(lambda c: bool(ERR.search(c))).any()
                                 and tools.command.fillna("").str.contains("test|pytest|npm|cargo|go ", regex=True).any()),
            "claims_completion": bool(DONE.search(last_resp[-400:])),
            "diff_chars": int(tools[tools.tool_name.isin(EDIT)].input_chars.sum()),
        }
        nxt = g.loc[bounds[k + 1]] if bounds[k + 1] is not None else None
        t_end = seg.timestamp.dropna().max() if seg.timestamp.notna().any() else (
            g.loc[bounds[k], "timestamp"] if bounds[k] is not None else pd.NaT)
        t_start = g.loc[bounds[k], "timestamp"] if bounds[k] is not None else pd.NaT
        q = qops[(qops.turn_number > (g.loc[bounds[k], "turn_number"] if bounds[k] is not None else -1)) &
                 (qops.turn_number < (nxt.turn_number if nxt is not None else 1e18))]
        rows.append({
            "session_id": g.session_id.iloc[0], "idx": len(rows),
            "prefix_user": json.dumps([h[:PREFIX_CHARS] for h in hist[-PREFIX_USER_N:]], ensure_ascii=False),
            "agent_last_response": last_resp[:PREFIX_CHARS],
            "agent_thinking_snippet": (think.content.iloc[-1] or "")[:300] if len(think) else "",
            "tools_used": json.dumps(tools.tool_name.value_counts().to_dict()),
            "files_touched": json.dumps(sorted(set(tools.file_path.dropna()))[:20]),
            "first_write_seen": edits_seen > 0,
            **f, "agent_turn_type": turn_type(f),
            "real_action": "no_op" if nxt is None else "message",
            "real_message": None if nxt is None else nxt.content,
            # v1 behaviour label: the dataset's own prompt_pushback label, no tagger of ours
            "behaviour": "no_op" if nxt is None else (nxt.prompt_pushback if isinstance(nxt.prompt_pushback, str) else "unlabeled"),
            "prompt_intent": None if nxt is None else nxt.prompt_intent,
            "prompt_pushback": None if nxt is None else nxt.prompt_pushback,
            "agent_turn_s": (t_end - t_start).total_seconds() if pd.notna(t_end) and pd.notna(t_start) else None,
            "user_latency_s": (nxt.timestamp - t_end).total_seconds() if nxt is not None and pd.notna(nxt.timestamp) and pd.notna(t_end) else None,
            "queued_msgs": int((q.queue_op_subtype == "user_prompt_enqueued").sum()),
            "first_prompt": hist[0][:2000] if hist else (nxt.content[:2000] if nxt is not None else ""),
        })
        if nxt is not None: hist.append(nxt.content or "")
    return rows


if __name__ == "__main__":
    assert is_injected("[Request interrupted by user]") and is_injected("[Image: image/png]") and is_injected("<command-name>/exit</command-name>")
    assert not is_injected("yes") and not is_injected("[Image: image/png] why is this broken?")
    c = pd.read_parquet(D / "processed/conv_slim.parquet")
    out = []
    for sid, g in c.groupby("session_id", sort=False):
        out += session_checkpoints(g)
    cp = pd.DataFrame(out)
    cp["checkpoint_id"] = cp.session_id + "#t" + cp.idx.astype(str)
    cp.to_parquet(D / "processed/checkpoints.parquet", index=False)
    print(cp.shape, cp.agent_turn_type.value_counts().to_dict(), cp.real_action.value_counts().to_dict())
