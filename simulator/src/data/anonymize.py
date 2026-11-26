"""Regex masking of obvious secrets/identifiers (§2.5). shortcut: regex only, names in free text survive;
add an NER/PII model before any data or fine-tuned weights leave this machine."""
import re

SUBS = [(r"[\w.+-]+@[\w-]+\.[\w.]+", "<EMAIL>"),
        (r"\b(sk|ghp|gho|xox[bap]|AKIA|hf)[_-]?[A-Za-z0-9\-_]{16,}", "<TOKEN>"),
        (r"\b[A-Za-z0-9+=_-]{40,}\b", "<TOKEN>"),
        (r"(/Users/|/home/|[A-Za-z]:\\[Uu]sers\\)[^/\\\s\"']+", r"\1<USER>"),
        (r"https?://\S+", "<URL>")]


def anon(t):
    for p, r in SUBS:
        t = re.sub(p, r, t or "")
    return t


if __name__ == "__main__":
    out = anon(r"mail a.b@x.io key ghp_abcdefghij0123456789 at /home/bob/x C:\Users\bob\y")
    assert "a.b@x.io" not in out and "ghp_" not in out and "bob" not in out, out
    print("ok")
