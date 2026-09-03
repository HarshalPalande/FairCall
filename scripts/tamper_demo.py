"""
Demonstrates that the audit log's hash chain actually detects tampering.

`make demo` only ever shows the happy path ("chain verified OK"), which is the
least convincing thing a hash chain can do — a log with no chaining at all also
passes that. This script writes a few decisions, verifies, then edits one past
entry the way someone covering their tracks would, and verifies again.

    python -m scripts.tamper_demo      (or: make tamper-demo)

Runs against a scratch log file, never the real audit_log/decisions.jsonl.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit import log_decision, verify_chain

DECISIONS = [
    {"transaction_id": "TXN-001", "action": "AUTO_CONTEST", "amount": 1_800.0, "win_probability": 0.93},
    {"transaction_id": "TXN-002", "action": "ESCALATE", "amount": 4_500.0, "win_probability": 0.17},
    {"transaction_id": "TXN-003", "action": "ESCALATE", "amount": 42_000.0, "win_probability": 0.64},
]

RULE = "=" * 78


def show(log_path, label):
    ok, msg = verify_chain(log_path)
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: {msg}")
    return ok


def main():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "decisions.jsonl"

        print(RULE)
        print("AUDIT LOG TAMPER DETECTION")
        print(RULE)

        print("\n1. Logging 3 decisions to a hash-chained, append-only log...")
        for d in DECISIONS:
            entry = log_decision(d, log_path=log_path)
            print(f"     {d['transaction_id']}  {d['action']:<13} entry_hash={entry['entry_hash'][:16]}...")

        print()
        show(log_path, "chain as written")

        print("\n2. Tampering: rewriting TXN-003 to look like it was auto-contested")
        print("   for a smaller amount — exactly what someone hiding a bad call would do.")
        lines = log_path.read_text().splitlines()
        entry = json.loads(lines[2])
        print(f"     before:  action={entry['body']['action']}  amount=₹{entry['body']['amount']:,.2f}")
        entry["body"]["action"] = "AUTO_CONTEST"
        entry["body"]["amount"] = 9_000.0
        lines[2] = json.dumps(entry)
        log_path.write_text("\n".join(lines) + "\n")
        print(f"     after:   action={entry['body']['action']}  amount=₹{entry['body']['amount']:,.2f}")

        print()
        tampered_ok = show(log_path, "chain after tamper")

        print("\n3. Trying to cover it up by appending more valid decisions on top...")
        log_decision(
            {"transaction_id": "TXN-004", "action": "ESCALATE", "amount": 7_000.0}, log_path=log_path
        )
        print()
        healed_ok = show(log_path, "chain after appending")

        print()
        print(RULE)
        if not tampered_ok and not healed_ok:
            print("RESULT: tampering detected, and appending does not heal it.")
            print("        Each entry hashes sha256(prev_hash + body), so editing any past")
            print("        entry breaks every link after it. A per-entry hash with no")
            print("        prev_hash would have passed step 2 — this is why it's chained.")
        else:  # pragma: no cover - only reachable if the chain is broken
            print("RESULT: FAILED — tampering was NOT detected. The audit claim is broken.")
            sys.exit(1)
        print(RULE)


if __name__ == "__main__":
    main()
