# AI Risk Manager — full-scope design

**Razorpay AI Buildathon, Track 02.** Reason code Visa **13.1** ("Merchandise /
Services Not Received").

> **What this document is.** The full-scope design for this system — including the
> parts that are *not* built. `README.md` documents what actually ships and is
> tested; this document is the north star it was cut down from, and it exists so
> the cuts are legible as decisions rather than as gaps. Written as a consolidated
> design record, not as a dated historical artifact — where it describes something
> unbuilt, it says so in the same breath.

---

## 1. The problem

A chargeback is a customer asking their issuing bank to reverse a payment. For
reason code 13.1 the claim is specifically "I never received what I paid for."
The merchant can either **accept** (eat the loss) or **contest** (submit an
evidence packet and hope to win).

Merchants lose contestable disputes for four reasons, and they are different
problems wearing the same costume:

| Root cause | What it actually is | Tractable? |
|---|---|---|
| Missing evidence | The proof exists but nobody attached it | Yes — rules |
| Weak / inconsistent evidence | The packet contradicts itself, or proves the wrong thing | Yes — rules |
| Late filing | The response window closed | Yes — dates |
| Genuine fraud / ATO | The customer really didn't get it, or is a chronic abuser | No — should not be contested |

The useful insight: **three of those four are not machine-learning problems.**
They are checklist problems. An ML model is needed for one thing only — deciding
whether *this particular* dispute, with *this* evidence, is worth the cost of
contesting. That framing drives the entire architecture below.

## 2. Design principles

1. **The litmus test.** If the ML model disappeared tomorrow, would the product
   still be useful? For this system, yes — the Evidence Completeness Checker is
   pure rules and carries real value alone. Anything that fails this test is a
   demo, not a product.
2. **Decisions in money, not confidence.** A model that says "78% likely to win"
   has not made a decision. Expected ₹ value has. See §4.
3. **Bounded autonomy.** The system's action space is two values. It can
   recommend contesting, or it can escalate to a human. It cannot submit, cannot
   contact a customer, and cannot exceed a hard ₹ ceiling regardless of how
   confident it is.
4. **Every number disclosed with its provenance.** Synthetic data is labelled
   synthetic. Assumed costs are labelled assumptions. A metric that flatters us
   and a metric that doesn't get the same font.
5. **Narrow and honest beats broad and shallow.** One reason code, done
   properly, over five done as a slide.

## 3. Architecture (full scope)

```
Razorpay payment.captured webhook
    │
    ▼
Transaction ──► Dispute Prevention Score  [BUILT]
                  ├── HIGH/MEDIUM ──► "collect evidence now" nudge to merchant
                  └── LOW          ──► standard monitoring

Razorpay payment.dispute.created webhook
    │
    ▼
Dispute ──► Feature Store (past-only UID aggregates)        [BUILT]
        ──► XGBoost detector ──► isotonic calibrator        [BUILT]
        ──► Evidence Completeness Checker (rules, no ML)    [BUILT]
        ──► Counterfactual engine (what evidence helps?)    [BUILT]
        ──► EV Decision Engine (cost table + hard ceiling)  [BUILT]
              ├── AUTO_CONTEST ──► Grounded LLM draft       [NOT BUILT — §6]
              │                     └── human review gate   [NOT BUILT]
              └── ESCALATE     ──► Human Review Queue       [BUILT]
        ──► Hash-chained audit log                          [BUILT]
        ──► SHAP explainability                             [BUILT]
```

## 4. The EV Decision Engine

The core of the system, and the piece most likely to be replaced by a confidence
threshold in a weaker build.

```
EV(accept)  = −amount
EV(contest) = amount × (2·P(win) − 1) − contest_cost
```

Auto-contest requires **all three**:

1. `EV(contest) > EV(accept)`
2. `amount ≤ HARD_CEILING_INR` — fixed, never overridden by model confidence
3. `evidence_completeness ≥ MIN_EVIDENCE_COMPLETENESS_FOR_AUTO`

**A nuance worth stating because it is easy to get wrong.** Because the upside
scales with `amount` while `contest_cost` is fixed, this EV formula *never*
disagrees with a naive "contest if P(win) > 50%" rule — higher stakes make
contesting more attractive under risk-neutral EV, not less. The real divergence
from a naive threshold comes from the **hard ceiling**, which is a deliberate
risk-aversion rule sitting *outside* the EV math. It exists because a business
rationally wants to be risk-averse on any single large automated decision even
when risk-neutral EV says proceed. Claiming the EV formula itself produces that
divergence would be a misrepresentation, so we don't.

## 5. Data strategy, and why it is synthetic

Considered and rejected: reframing IEEE-CIS as chargeback outcomes. It is a
*fraud-detection* dataset with no win/loss labels for contested disputes; using
it would mean presenting a proxy as ground truth silently.

Chosen: a fully synthetic generator with a disclosed data-generating process and
a fixed seed. It buys reproducibility, and it lets the four root causes in §1 be
encoded as real recoverable structure. It costs external validity, and **every
metric in this project is a metric against synthetic ground truth** — stated
wherever a number is reported, not once in a footnote.

The honest cost: the generator couples customer trust to evidence availability
(`evidence_base_rate = 0.35 + 0.5 × trust`), so `cust_prior_win_rate` and the
evidence flags are correlated *by construction*. SHAP ranks customer history
first and the three evidence flags immediately behind. That is a property of our
generator, not a discovered fact about chargebacks — see README.

## 6. Not built, and why

- **Grounded LLM evidence generator — now prototyped** (`src/evidence_draft.py`).
  The brief allows detector *or* verifier *or* auto-responder; the core system is
  detector + verifier and contains no LLM. The responder exists as a
  clearly-labelled prototype: a fact ledger built only from evidence on file,
  generation constrained to cited claims (the letter's boilerplate is our code,
  not the model's), and a pure-Python verifier that rejects rather than repairs a
  draft containing a fabricated citation, an uncited claim, or an assertion the
  ledger cannot support. Human-gated; nothing is submitted anywhere. What remains
  unbuilt is **semantic entailment** — a claim citing a real fact while
  overstating it still passes — and wiring the draft into the EV decision path.
- **Merchant Intelligence Hub, continuous retraining.** Phase 3 vision items.
- **Multiple reason codes.** Deliberate. The rule table in `src/evidence.py`
  generalises by turning `REQUIRED_EVIDENCE_FOR_13_1` into a per-code dict; the
  reason it hasn't been done is that one slice done honestly was the goal.
- **Autonomous submission.** Not a scope cut — a design boundary. `AUTO_CONTEST`
  means *recommend*, and a production build keeps a human gate before submission.

## 7. Roadmap

**Phase 1 — prove the loop.** Detector → evidence → EV → escalate/auto → audit,
measured honestly on one reason code. *(This is what ships.)*

**Phase 2 — real signal.** Replace synthetic ground truth with real dispute
outcomes via `payment.dispute.created`, accumulating labels before retraining
anything. Move the cost table from disclosed placeholders to numbers owned by
finance/risk ops, likely varying by merchant category. This is the biggest gap
between this repo and a production system, and no amount of Phase-1 polish
substitutes for it.

**Phase 3 — automate more, carefully.** Raise the auto-contest rate without
raising false-positive cost disproportionately: shadow mode first, then staged
rollout (1% → 10% → 100%) gated on the FP-cost metric, with a kill switch. The
hash-chained audit log stops being a demo feature and becomes what compliance
queries during a card-network review.
