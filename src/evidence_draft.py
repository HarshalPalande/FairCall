"""
Grounded evidence-draft step — PROTOTYPE, human-gated, never auto-submitted.

This is the one piece the rest of the repo deliberately cut (see buildathon.md
§6). It is included as a clearly-labelled prototype rather than a production
component, and it is built so that the LLM is the *least* trusted part of it.

THE DESIGN PROBLEM
    An LLM asked to "write a chargeback rebuttal" will write a fluent, confident
    letter asserting the parcel was delivered and signed for — whether or not any
    of that is true. In a dispute response, a fabricated claim is not a bad
    sentence; it is a false representation to a card network. So the question is
    not "can the model write well" but "can the system prove the model didn't
    make anything up."

THE APPROACH — constrain, then verify
    1. FACT LEDGER (`build_fact_ledger`) — we build a closed set of atomic facts
       from the dispute, and it contains ONLY evidence actually on file. Absent
       evidence produces no fact. The model never sees the raw case, only this
       ledger.
    2. CONSTRAINED GENERATION — the model may return exactly one thing: a list of
       claims, each carrying citations to ledger fact IDs. It does not write the
       letter. Every non-claim word in the final document — greeting, structure,
       sign-off — is boilerplate from `render_letter()` in this file. Minimising
       the model's output surface is what makes the output checkable.
    3. VERIFICATION (`verify_draft`) — pure Python, no model, no network. A draft
       that cites a non-existent fact, makes an uncited claim, or asserts
       something the ledger cannot support is REJECTED, not repaired.
    4. HUMAN GATE — an approved draft is still only a draft. Nothing here submits
       anything anywhere, and `Action` remains a two-value enum elsewhere in the
       system.

HONEST LIMITATION
    `verify_draft` checks citation validity and scans for assertions whose
    supporting evidence type is absent from the ledger. It does NOT perform
    semantic entailment: a claim that cites a real fact but overstates what that
    fact shows would pass. Closing that gap properly needs a second-pass
    entailment check (a natural next step, not built). Stated here rather than
    left for a reviewer to find — the keyword scan is a floor, not a proof.

The ledger and the verifier are pure functions with no API dependency, so they
are unit-tested without a key and remain useful even if the model call is turned
off entirely — the same litmus test `src/evidence.py` is built around.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from src import config
from src.evidence import CompletenessResult

# Only used when actually calling the model. Kept out of module import so the
# ledger + verifier work with no SDK installed and no key set.
MODEL = "claude-opus-5"
MAX_TOKENS = 4_000

# Human-readable statement for each evidence type, used to build the ledger.
EVIDENCE_STATEMENTS = {
    "tracking_number": "A carrier tracking number is on file for this shipment.",
    "delivery_confirmation": "The carrier recorded this shipment as delivered.",
    "signed_pod": "A signed proof of delivery was captured at the delivery address.",
    "courier_communication": "Courier correspondence about this shipment is on file.",
    "avs_match": "The billing address passed Address Verification (AVS) at purchase.",
}

# Assertions that require a specific evidence type to be present. If a claim's
# text trips one of these patterns and the corresponding evidence is NOT in the
# ledger, the claim is unsupported regardless of what it cites. This catches the
# subtle failure the citation check alone misses: a real citation attached to a
# sentence making a broader claim than the ledger supports.
UNSUPPORTED_CLAIM_PATTERNS = {
    "signed_pod": re.compile(r"\bsign(?:ed|ature)\b|\bproof of delivery\b|\bPOD\b", re.I),
    "delivery_confirmation": re.compile(r"\bdelivered\b|\bdelivery was confirmed\b|\breceipt confirmed\b", re.I),
    "tracking_number": re.compile(r"\btracking (?:number|id)\b|\btracked\b", re.I),
    "avs_match": re.compile(r"\bAVS\b|\baddress verification\b", re.I),
}

SYSTEM_PROMPT = """You draft factual claims for a payment-dispute rebuttal (Visa reason code 13.1, "merchandise or services not received").

You will be given a numbered FACT LEDGER. It is the complete and only set of facts available about this dispute.

Rules, all strictly enforced by an automated verifier that will reject your output:
1. Every claim you write must be supported by one or more ledger facts, and must cite them by ID.
2. You may not state, imply, or hint at anything the ledger does not contain. If the ledger does not say a parcel was delivered, you may not say or imply it was.
3. Absence of a fact is not evidence. Never explain away missing evidence, never speculate about what probably happened, and never characterise the cardholder.
4. Write plainly and neutrally, as a factual submission to a card network. No persuasion, no adjectives of emphasis, no appeals.
5. If the ledger is too thin to support a rebuttal, return an empty claims list. That is a valid and correct answer.

Return only claims. Do not write a greeting, a letter, or a closing — those are added by the system."""

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "cites": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "cites"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Fact:
    id: str
    source: str
    statement: str


@dataclass
class Claim:
    text: str
    cites: list


@dataclass
class VerificationResult:
    approved: bool
    violations: list = field(default_factory=list)
    verified_claims: list = field(default_factory=list)


@dataclass
class DraftResult:
    status: str  # APPROVED_FOR_HUMAN_REVIEW | REJECTED | UNAVAILABLE
    ledger: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    verification: VerificationResult = None
    letter: str = ""
    detail: str = ""


def build_fact_ledger(raw_case: dict, evidence_result: CompletenessResult) -> list[Fact]:
    """Closed set of atomic facts for this dispute.

    Evidence that is absent produces NO fact — the model is never given the
    opportunity to reason about, or write around, a gap. Transaction facts are
    included because they are objective properties of the payment record.
    """
    facts: list[Fact] = []

    def add(source: str, statement: str):
        facts.append(Fact(id=f"F{len(facts) + 1}", source=source, statement=statement))

    add("transaction.amount", f"The disputed transaction is for ₹{raw_case['amount_inr']:,.2f}.")
    add("transaction.category", f"The order was placed in the {raw_case['merchant_category']} category.")
    add("transaction.shipping", f"The order was fulfilled by {raw_case['shipping_method'].replace('_', ' ')}.")

    if raw_case.get("account_age_days_at_dispute") is not None:
        add(
            "customer.account_age",
            f"The customer account was {int(raw_case['account_age_days_at_dispute'])} days old at the time of the dispute.",
        )

    for etype in config.EVIDENCE_TYPES:
        if raw_case.get(f"has_{etype}", 0) == 1:
            add(f"evidence.{etype}", EVIDENCE_STATEMENTS[etype])

    if raw_case.get("late_filing", 0) == 1:
        add("dispute.timing", "The dispute was filed more than 30 days after the transaction date.")

    return facts


def render_ledger(facts: list[Fact]) -> str:
    return "\n".join(f"[{f.id}] {f.statement}" for f in facts)


def _present_evidence_types(facts: list[Fact]) -> set:
    return {f.source.split("evidence.", 1)[1] for f in facts if f.source.startswith("evidence.")}


def verify_draft(claims: list[Claim], facts: list[Fact]) -> VerificationResult:
    """Pure, deterministic verification. No model, no network.

    Rejects — never repairs — a draft that:
      * cites a fact ID that does not exist (hallucinated citation),
      * makes a claim with no citation at all,
      * asserts something whose supporting evidence type is absent from the
        ledger, even if the claim carries an otherwise-valid citation.
    """
    valid_ids = {f.id for f in facts}
    present = _present_evidence_types(facts)
    violations: list[str] = []
    verified: list[Claim] = []

    for i, claim in enumerate(claims):
        claim_ok = True
        text = claim.text.strip()

        if not text:
            violations.append(f"claim {i}: empty claim text")
            continue

        if not claim.cites:
            violations.append(f"claim {i}: no citation — every claim must cite at least one fact")
            claim_ok = False

        for cite in claim.cites:
            if cite not in valid_ids:
                violations.append(
                    f"claim {i}: cites {cite}, which is not in the fact ledger (fabricated citation)"
                )
                claim_ok = False

        for etype, pattern in UNSUPPORTED_CLAIM_PATTERNS.items():
            if pattern.search(text) and etype not in present:
                violations.append(
                    f"claim {i}: asserts '{etype.replace('_', ' ')}' but no such evidence is on file — unsupported claim"
                )
                claim_ok = False

        if claim_ok:
            verified.append(claim)

    return VerificationResult(
        approved=not violations and bool(verified),
        violations=violations,
        verified_claims=verified,
    )


def render_letter(claims: list[Claim], raw_case: dict) -> str:
    """Boilerplate is OURS, not the model's. The model contributed only the
    cited claim sentences below."""
    body = "\n".join(
        f"  {i}. {c.text.strip()}  [{', '.join(c.cites)}]" for i, c in enumerate(claims, 1)
    )
    return (
        f"RE: Dispute response — transaction {raw_case.get('transaction_id', 'UNKNOWN')}\n"
        f"Reason code {config.REASON_CODE} (merchandise / services not received)\n\n"
        "The merchant submits the following factual statements, each supported by\n"
        "documentation on file and referenced by evidence ID:\n\n"
        f"{body}\n\n"
        "Supporting documentation is attached. The merchant requests that the\n"
        "dispute be reviewed against the evidence cited above.\n\n"
        "--- DRAFT — REQUIRES HUMAN REVIEW BEFORE ANY SUBMISSION ---"
    )


def draft_contest_response(raw_case: dict, evidence_result: CompletenessResult) -> DraftResult:
    """Generate a grounded draft, verify it, and gate it behind human review.

    Returns UNAVAILABLE (never raises, never fabricates) if the SDK or API key
    is missing, so the rest of the pipeline is unaffected by this being a
    prototype. A REJECTED result deliberately carries no letter text — a draft
    that failed verification is not offered for use in any form.
    """
    facts = build_fact_ledger(raw_case, evidence_result)

    if not facts:
        return DraftResult(status="REJECTED", ledger=facts, detail="empty fact ledger — nothing to assert")

    try:
        import anthropic
    except ImportError:
        return DraftResult(
            status="UNAVAILABLE", ledger=facts,
            detail="anthropic SDK not installed — run `pip install anthropic`. Ledger and verifier still work.",
        )

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return DraftResult(
            status="UNAVAILABLE", ledger=facts,
            detail="no ANTHROPIC_API_KEY set — ledger and verifier are unaffected and still run.",
        )

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"FACT LEDGER\n{render_ledger(facts)}"}],
            output_config={"format": {"type": "json_schema", "schema": CLAIMS_SCHEMA}},
        )
    except anthropic.RateLimitError as e:
        return DraftResult(status="UNAVAILABLE", ledger=facts, detail=f"rate limited: {e}")
    except anthropic.APIStatusError as e:
        return DraftResult(status="UNAVAILABLE", ledger=facts, detail=f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError as e:
        return DraftResult(status="UNAVAILABLE", ledger=facts, detail=f"connection error: {e}")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return DraftResult(status="REJECTED", ledger=facts, detail="model returned unparseable output")

    claims = [Claim(text=c.get("text", ""), cites=list(c.get("cites", []))) for c in payload.get("claims", [])]

    if not claims:
        return DraftResult(
            status="REJECTED", ledger=facts, claims=[],
            detail="model returned no claims — the evidence on file does not support a rebuttal",
        )

    verification = verify_draft(claims, facts)
    if not verification.approved:
        return DraftResult(
            status="REJECTED", ledger=facts, claims=claims, verification=verification,
            detail=f"{len(verification.violations)} verification violation(s); draft withheld",
        )

    return DraftResult(
        status="APPROVED_FOR_HUMAN_REVIEW",
        ledger=facts,
        claims=verification.verified_claims,
        verification=verification,
        letter=render_letter(verification.verified_claims, raw_case),
        detail="passed citation and unsupported-assertion checks; requires human review before submission",
    )
