"""
Tests for the hash-chained audit log (src/audit.py).

The README claims this chain is "an actual hash chain, not a decoration" — that
it detects tampering with, reordering of, or deletion of any past entry. These
tests are that claim, executed. The happy path (`chain verified OK`) is the
least interesting thing here; the tamper cases are the point, because a hash of
only the current entry would pass the happy path and fail every one of them.
"""
import json

from src.audit import GENESIS_HASH, log_decision, verify_chain


def _write_entries(path, n=3):
    for i in range(n):
        log_decision({"transaction_id": f"T{i}", "action": "ESCALATE", "amount": 1000 + i}, log_path=path)


def test_empty_log_verifies(tmp_path):
    ok, msg = verify_chain(tmp_path / "does_not_exist.jsonl")
    assert ok is True
    assert "empty" in msg


def test_untampered_chain_verifies(tmp_path):
    log = tmp_path / "decisions.jsonl"
    _write_entries(log, 5)
    ok, msg = verify_chain(log)
    assert ok is True, msg
    assert msg == "chain verified OK"


def test_first_entry_links_to_genesis(tmp_path):
    log = tmp_path / "decisions.jsonl"
    _write_entries(log, 1)
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["prev_hash"] == GENESIS_HASH


def test_each_entry_links_to_the_previous_one(tmp_path):
    log = tmp_path / "decisions.jsonl"
    _write_entries(log, 4)
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    for prev, curr in zip(entries, entries[1:]):
        assert curr["prev_hash"] == prev["entry_hash"]


def test_tampered_body_is_detected(tmp_path):
    """Edit a past decision's amount — the recomputed hash must stop matching."""
    log = tmp_path / "decisions.jsonl"
    _write_entries(log, 3)

    lines = log.read_text().splitlines()
    entry = json.loads(lines[1])
    assert entry["body"]["amount"] == 1001
    entry["body"]["amount"] = 999_999  # the tamper
    lines[1] = json.dumps(entry)
    log.write_text("\n".join(lines) + "\n")

    ok, msg = verify_chain(log)
    assert ok is False
    assert "line 1" in msg
    assert "tampered" in msg


def test_deleted_entry_is_detected(tmp_path):
    """Drop a line entirely. A per-entry hash with no prev_hash would NOT catch
    this — the surviving entries are each individually still valid."""
    log = tmp_path / "decisions.jsonl"
    _write_entries(log, 4)

    lines = log.read_text().splitlines()
    del lines[1]
    log.write_text("\n".join(lines) + "\n")

    ok, msg = verify_chain(log)
    assert ok is False
    assert "chain broken or reordered" in msg


def test_reordered_entries_are_detected(tmp_path):
    """Swap two entries. Same argument as deletion: each entry's own hash is
    still internally consistent, so only the prev_hash linkage catches it."""
    log = tmp_path / "decisions.jsonl"
    _write_entries(log, 4)

    lines = log.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    log.write_text("\n".join(lines) + "\n")

    ok, msg = verify_chain(log)
    assert ok is False
    assert "chain broken or reordered" in msg


def test_appending_after_a_tamper_still_fails(tmp_path):
    """A tamper cannot be 'healed' by writing more entries on top of it —
    the broken link stays broken no matter what follows."""
    log = tmp_path / "decisions.jsonl"
    _write_entries(log, 2)

    lines = log.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["body"]["action"] = "AUTO_CONTEST"  # rewrite history
    lines[0] = json.dumps(entry)
    log.write_text("\n".join(lines) + "\n")

    log_decision({"transaction_id": "T99", "action": "ESCALATE", "amount": 5000}, log_path=log)

    ok, msg = verify_chain(log)
    assert ok is False
    assert "line 0" in msg
