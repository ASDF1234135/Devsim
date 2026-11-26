# Codebook v0 (DRAFT, not yet human-validated)

Status: DEFERRED extension (plan v0.8, §6). v1 uses the §2.2 behaviour labels; nothing in v1 reads this file. Not human-validated.

## Action (what the user does at a checkpoint)
| Action | Definition | Edge cases |
|---|---|---|
| no_op | No user message before the next agent turn / session end | Queued-then-discarded messages count as no_op |
| answer | Replies to an agent question | Mixed answer + new ask → `new_requirement` if the ask is the main content |
| ask_clarification | Asks the agent to explain or clarify | |
| new_requirement | Adds scope not in the goal | |
| redirect | Changes approach for existing scope | Corrections of agent errors (`prompt_pushback=correction`) |
| approve / accept | Agrees to a plan / the agent's completion claim | "ok", "lgtm", "continue" |
| deny / reject | Refuses a plan / a completion claim | `prompt_pushback=rejection` |
| inspect / request_evidence / check_external | Asks for diff, test output, external check | "run the tests" = check_external |
| defer / dont_know | Postpones or lacks the information | |

Seed from data: SWE-chat `prompt_intent` (understand, git, debug, create new code, refactor, test, connect, other) and `prompt_pushback` (non_pushback, correction, failure_report, rejection) are coarse LLM labels, usable as weak labels only.

## Codes (v1 coverage per plan §2.2)
D disclosure: full. I intent: full (new/revised goal, goal id). C channel/timing: limited (chat; before/after first write). R review: coarse accept vs correct. K knowledge: subset, `unknown` separate. B, A: not annotated (fixtures / simulator state). P: partial. E: time model only.

Unknown handling: any message that cannot be decided from the transcript gets `unknown`, never a guess.
