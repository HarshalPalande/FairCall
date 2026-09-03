"""
Tests for the grounded evidence-draft prototype (src/evidence_draft.py).

These test the two parts that must be trustworthy: the fact ledger (what the
model is allowed to know) and the verifier (what the model is allowed to say).
Neither touches the network or needs an API key — which is the point. The LLM is
the untrusted component here; the guardrails around it are pure functions, so
they can be tested exhaustively and cheaply.

The adversarial cases below are written as "what would a hallucinating model
produce", because that is the actual threat model for this component.
"""
import pytest

from src import config
from src.evidence import evaluate_evidence
from src.evidence_draft import (
    Claim,
    build_fact_ledger,
    draft_contest_response,
    render_ledger,
    render_letter,
    verify_draft,
)

FULL_CASE = {
    "transaction_id": "TXN-TEST",
    "amount_inr": 4_200.0,
    "merchant_category": "electronics",
    "shipping_method": "standard",
    "account_age_days_at_dispute": 200,
    "late_filing": 0,
    **{f"has_{e}": 1 for e in config.EVIDENCE_TYPES},
}

BARE_CASE = {
    "transaction_id": "TXN-BARE",
    "amount_inr": 4_200.0,
    "merchant_category": "electronics",
    "shipping_method": "standard",
    "account_age_days_at_dispute": 12,
    "late_filing": 1,
    **{f"has_{e}": 0 for e in config.EVIDENCE_TYPES},
}


def _ledger(case):
    ev = evaluate_evidence(
        {e: case.get(f"has_{e}", 0) for e in config.EVIDENCE_TYPES}, case["shipping_method"]
    )
    return build_fact_ledger(case, ev)


# --- the fact ledger: what the model is allowed to know ---------------------


def test_ledger_contains_a_fact_per_present_evidence_type():
    facts = _ledger(FULL_CASE)
    sources = {f.source for f in facts}
    for etype in config.EVIDENCE_TYPES:
        assert f"evidence.{etype}" in sources


def test_ledger_omits_absent_evidence_entirely():
    """The model must never be handed a gap it could write around."""
    facts = _ledger(BARE_CASE)
    assert not [f for f in facts if f.source.startswith("evidence.")]


def test_ledger_ids_are_unique_and_sequential():
    facts = _ledger(FULL_CASE)
    assert [f.id for f in facts] == [f"F{i}" for i in range(1, len(facts) + 1)]


def test_ledger_records_late_filing_against_the_merchant():
    """An honest ledger includes facts that don't help the merchant."""
    facts = _ledger(BARE_CASE)
    assert any(f.source == "dispute.timing" for f in facts)
    assert not any(f.source == "dispute.timing" for f in _ledger(FULL_CASE))


def test_render_ledger_is_id_prefixed():
    facts = _ledger(FULL_CASE)
    rendered = render_ledger(facts)
    assert rendered.startswith("[F1]")
    assert all(f"[{f.id}]" in rendered for f in facts)


# --- the verifier: what the model is allowed to say -------------------------


def _fact_id(facts, source):
    return next(f.id for f in facts if f.source == source)


def test_well_grounded_draft_is_approved():
    facts = _ledger(FULL_CASE)
    claims = [
        Claim(
            "The carrier recorded this shipment as delivered.",
            [_fact_id(facts, "evidence.delivery_confirmation")],
        ),
        Claim(
            "A signed proof of delivery was captured.",
            [_fact_id(facts, "evidence.signed_pod")],
        ),
    ]
    result = verify_draft(claims, facts)
    assert result.approved is True, result.violations
    assert result.violations == []
    assert len(result.verified_claims) == 2


def test_fabricated_citation_is_rejected():
    """The classic hallucination: a confident claim citing a fact that does
    not exist."""
    facts = _ledger(FULL_CASE)
    claims = [Claim("The parcel was delivered to the cardholder.", ["F99"])]
    result = verify_draft(claims, facts)

    assert result.approved is False
    assert any("F99" in v and "fabricated citation" in v for v in result.violations)
    assert result.verified_claims == []


def test_uncited_claim_is_rejected():
    facts = _ledger(FULL_CASE)
    claims = [Claim("The customer has received the goods.", [])]
    result = verify_draft(claims, facts)

    assert result.approved is False
    assert any("no citation" in v for v in result.violations)


def test_assertion_without_supporting_evidence_is_rejected():
    """The subtle failure the citation check alone misses: a REAL citation
    attached to a claim asserting more than the ledger holds. Here the case has
    no signed POD, but the model asserts one while citing a valid fact."""
    facts = _ledger(BARE_CASE)
    valid_id = facts[0].id
    claims = [Claim("The delivery was signed for by the cardholder.", [valid_id])]

    result = verify_draft(claims, facts)
    assert result.approved is False
    assert any("unsupported claim" in v for v in result.violations)


@pytest.mark.parametrize(
    "text,evidence_type",
    [
        ("A signature was captured on delivery.", "signed_pod"),
        ("The order was delivered on time.", "delivery_confirmation"),
        ("The tracking number confirms dispatch.", "tracking_number"),
        ("AVS matched at the time of purchase.", "avs_match"),
    ],
)
def test_each_unsupported_assertion_pattern_is_caught(text, evidence_type):
    facts = _ledger(BARE_CASE)
    result = verify_draft([Claim(text, [facts[0].id])], facts)
    assert result.approved is False
    assert any(evidence_type.replace("_", " ") in v for v in result.violations)


def test_same_assertion_is_allowed_when_evidence_is_present():
    """The scan must not be a blanket keyword ban — with the evidence on file,
    the identical sentence is legitimate."""
    facts = _ledger(FULL_CASE)
    pod_fact = next(f for f in facts if f.source == "evidence.signed_pod")
    result = verify_draft([Claim("The delivery was signed for.", [pod_fact.id])], facts)
    assert result.approved is True, result.violations


def test_empty_claim_list_is_not_approved():
    """No claims is a valid model answer, but it is not an approved draft."""
    result = verify_draft([], _ledger(FULL_CASE))
    assert result.approved is False


def test_one_bad_claim_fails_the_whole_draft():
    """A draft is rejected, not silently repaired by dropping bad claims."""
    facts = _ledger(FULL_CASE)
    claims = [
        Claim(
            "The carrier recorded this shipment as delivered.",
            [_fact_id(facts, "evidence.delivery_confirmation")],
        ),
        Claim("The cardholder has a history of fraudulent claims.", ["F404"]),
    ]
    result = verify_draft(claims, facts)
    assert result.approved is False


# --- rendering + the human gate ---------------------------------------------


def test_letter_carries_citations_and_a_human_review_marker():
    facts = _ledger(FULL_CASE)
    delivered = _fact_id(facts, "evidence.delivery_confirmation")
    claims = [Claim("The carrier recorded this shipment as delivered.", [delivered])]
    letter = render_letter(claims, FULL_CASE)

    assert f"[{delivered}]" in letter
    assert "TXN-TEST" in letter
    assert "REQUIRES HUMAN REVIEW" in letter
    assert config.REASON_CODE in letter


def test_draft_is_unavailable_without_credentials_not_fabricated(monkeypatch):
    """With no API key the prototype must degrade to UNAVAILABLE — never invent
    a letter, and never break the pipeline around it."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    ev = evaluate_evidence(
        {e: FULL_CASE.get(f"has_{e}", 0) for e in config.EVIDENCE_TYPES}, FULL_CASE["shipping_method"]
    )
    result = draft_contest_response(FULL_CASE, ev)

    assert result.status == "UNAVAILABLE"
    assert result.letter == ""
    assert result.ledger, "the ledger must still be built without a model"


def test_rejected_drafts_never_carry_letter_text():
    """A draft that failed verification must not be offered for use in any
    form — including as 'here's what it would have said'."""
    facts = _ledger(BARE_CASE)
    bad = verify_draft([Claim("It was signed for.", [facts[0].id])], facts)
    assert bad.approved is False
    # render_letter is only ever reached for approved drafts in
    # draft_contest_response; assert the contract holds there.
    from src.evidence_draft import DraftResult

    rejected = DraftResult(status="REJECTED", ledger=facts, verification=bad)
    assert rejected.letter == ""
