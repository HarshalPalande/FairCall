# How this got built — a walkthrough for a friend

This is a plain-language account of one working session on **AI Risk Manager**, a
hackathon submission for Razorpay's "Track 02" brief: build a working detector,
verifier, or auto-responder for one class of merchant loss — chargebacks, in this
case — with **honest metrics including false-positive cost**, and stay strictly
defense-only.

If you just want the pitch, read the README. This document is for understanding
*how* the project got to where it is and *why* the pieces look the way they do —
including the parts that went wrong before they went right, because that's most of
the interesting part.

---

## 1. What existed before this session

The project already had a working core: a chargeback outcome detector (an XGBoost
model predicting whether a merchant will *win* a dispute if they contest it), an
expected-value engine that decides contest-vs-accept, a tamper-evident hash-chained
audit log, and a prototype where an LLM drafts dispute-response letters but is only
allowed to cite facts from a pre-built evidence ledger — a pure-Python verifier
rejects anything it can't support.

The session picked up from there with one goal: make the project's actual
differentiators — the parts that go beyond "we trained a classifier" — visible,
provable, and honest.

---

## 2. The four-way decision engine, VAMP, CE3.0, and RDR

**The problem with the original engine.** It only had two outputs: contest a
dispute, or escalate it to a human. That's "fight or don't fight" — a real but
incomplete picture, because Visa's actual dispute infrastructure gives a merchant
more levers than that.

**What got added, and why each piece matters:**

- **VAMP (Visa's Acquirer Monitoring Program)** — every *filed* dispute counts
  against a merchant's portfolio-wide dispute ratio, whether the merchant wins or
  loses it. Cross a threshold (1.5%) and Visa applies a flat per-dispute penalty to
  the merchant's *entire* dispute volume, not just the one over the line. The
  original EV engine only priced the transaction amount — it had no idea this cost
  existed.
- **A four-way engine** (`src/decision_engine.py`): DEFLECT (push order proof to the
  card network before a dispute is even filed), AUTO_RESOLVE (refund the customer
  pre-dispute via Visa's RDR program), CONTEST (the original path), ESCALATE (human
  review). Each of these has a different VAMP consequence — DEFLECT and
  AUTO_RESOLVE, if they work, keep the event off the ratio entirely; CONTEST and
  ESCALATE don't, because the dispute was already filed.
- **The interesting consequence, and the whole reason this matters**: for a
  non-fraud dispute, refunding it via RDR — even one the merchant would probably
  *win* if they fought it — can be the cheaper decision, because winning protects
  the transaction value but does nothing for the portfolio ratio. A two-output
  engine literally cannot express this trade-off. This became the project's
  sharpest demo moment (see "the RDR moment" below).
- **CE3.0** (`src/ce3.py`) — a small, deterministic, zero-ML rules engine for a
  specific Visa program: if a merchant has two prior undisputed transactions from
  the same customer with matching device/IP data, a *fraud* dispute can be blocked
  pre-emptively, before it's even filed. No model, no judgment call — just a
  card-network rule coded exactly as written.
- **RDR Optimizer** (`src/rdr_optimizer.py`) — given a dispute distribution and the
  merchant's current VAMP headroom, what refund-threshold minimizes total cost?
  Sweeps the whole curve rather than just returning one number, so the trade-off
  (more refunds = more revenue given up, but more headroom bought back) is
  inspectable, not hidden behind a black-box recommendation.

**A safety bug caught in the design, not in production.** An earlier version of the
four-way engine scored ESCALATE against the other three options by expected value.
That's broken in a way that *passes tests that don't check the right thing*:
DEFLECT's expected value is a mix of "cheap success" and "the accept-loss," so it
mathematically dominates a flat accept-loss whenever deflection has *any* chance of
working — meaning a ₹42,000 over-ceiling dispute (one that's supposed to always go
to a human, no exceptions) would get silently routed to an *automated* action
instead. A test that only checks `chosen != CONTEST` goes green against this bug,
because the action genuinely isn't CONTEST — it's just a different automated
action. The fix: ESCALATE is never scored against the others. It's the fallback
when no automated action is permitted, not a competitor in the max() comparison.
That makes the "no large automated decision without a human" guarantee *structural*
— built into the code's shape — rather than something that happens to fall out of
arithmetic that could quietly stop being true.

All of this got wired directly into `make demo` — the one command a judge is told
to run — instead of living only in a Streamlit page nobody scripted a walkthrough
for. That mattered: before this, the project's best idea was reachable only by
clicking through a UI page.

---

## 3. Turning the argument into something you can touch

Text output convinces a reader who's already paying close attention. It doesn't
convince someone skimming fifty submissions. So two interactive pieces got built:

**The Four-Way Simulator** (`app/pages/8_Four_Way_Simulator.py`) — instead of
picking a fixed example, you drag sliders (amount, win probability, evidence
completeness, portfolio dispute volume) and watch both engines re-run live, side by
side. It's seeded to open *already disagreeing*: two-way engine says
`AUTO_CONTEST`, four-way engine says `AUTO_RESOLVE`, at 69% win probability — so the
disagreement is the first thing you see, not something you have to go looking for.
That's "the RDR moment" from above, made concrete: representment would probably
win, and the system refunds anyway, because the VAMP cost of a filed dispute
(~₹4,576 in this portfolio) gets priced onto CONTEST but not onto AUTO_RESOLVE.

Two real bugs got caught building this page, worth mentioning because they're the
kind of thing that's easy to ship by accident: a diverging bar chart meant to show
each option's expected value was scaled wrong — a bar was capped at a quarter of
its available width even at maximum magnitude, because a percentage got applied
twice (once for a half-width container, once for the fill fraction). And a
locale-dependent bug where a plain rupee amount rendered as `3100,00` — a
comma-decimal — because *any* float-typed number input renders that way in some
browser locales, regardless of whether the number itself has a fractional part.
Both were caught and fixed before they made it in front of anyone.

**VAMP gauges** on the VAMP Risk and RDR Optimizer pages — a simple color-graded
bar (green → amber → red) with a marker showing the current ratio against the
1.5% threshold. On the RDR Optimizer specifically, it shows *two* markers — before
and after the recommended refund policy — so the policy's actual effect reads as
movement on a bar, not two numbers you have to compare in your head.

---

## 4. Demo Mode — and the decision *not* to fake it

This is the part worth explaining most carefully, because it's a genuine ethical
fork in the road, not just an engineering choice.

There's a second model in this project — the **prevention model**
(`src/prevention.py`) — that scores a transaction *at payment time*, before any
dispute exists, trying to predict "will this become a dispute?" It's a much harder
problem than the outcome detector, because at payment time there's no evidence yet
(no tracking number, no delivery confirmation) — none of the signal the reactive
detector gets to use.

The original idea for a live demo was: simulate a bunch of "agents" submitting
payments, and show the system catching the risky ones. The tempting, dishonest
version of that demo rigs the numbers so it *always* catches everything — a clean,
impressive-looking, and completely fake result. That was explicitly rejected, for
two reasons:

1. **It would have contradicted the project's own disclosed metrics.** The
   prevention model's real recall (how much of the true risk it actually catches)
   is around 18–23% — genuinely modest, and *disclosed as such* everywhere else in
   the project. A demo claiming 100% would fall apart the moment anyone opened
   `artifacts/prevention_metrics.json`, and "our demo doesn't match our own numbers"
   is a worse look than a demo that's honestly imperfect.
2. **An adaptive "agent that learns to evade the detector" framing edges toward
   offense-capable** — and the track's rules are explicit: *"strictly defense-only;
   anything offense-capable is disqualified."* Even a harmless, synthetic version of
   that framing isn't worth the risk this close to submission.

**What got built instead** (`app/pages/9_Demo_Mode.py`): 100 fresh synthetic
transactions every click — no fixed seed, genuinely different each run — labeled
across 50 simulated "agent" session IDs for narrative flavor, scored live by the
*real* trained model. The page reports whatever actually happens, including the
misses — rows are color-tinted (green catches, red misses, amber false alarms),
and there are expandable "Caught" and "Missed, with the reason for each" sections
that state the honest, quantitative reason for a miss (how far the score fell short
of the threshold, and what the model actually saw) rather than inventing a
plausible-sounding cause.

Cases get bucketed into Easy / Moderate / Extreme difficulty *after* scoring, based
on how close the model's score sat to the decision boundary — not chosen in advance
to flatter the result. In a live run this showed exactly the honest shape you'd
expect: Easy cases caught almost every time, Extreme cases (the ones that
genuinely disguise themselves) caught almost never. That gradient is a *better*
story for a rigor-minded judge than a fake 100%, because it's falsifiable and it
matches the disclosed numbers.

A threshold slider was added later so a viewer can drag it live and watch recall
and false-alarm rate trade off against each other in real time (catch nearly
everything at a low threshold, at the cost of a lot of false alarms; catch almost
nothing at a high threshold, with very few false alarms). That turns "it misses a
lot" from a fixed weakness into an honest, interactive demonstration of a real
trade-off — which is a more sophisticated thing to show a judge than a single
number ever could be.

---

## 5. An outside QA pass, and what it found

Rather than trust only self-review, a full pass was run using **Claude for Chrome**
(a separate browser-based agent, with no memory of this session, given a detailed
written brief and told to click through every page, every control, and report back
in a structured "working / not working" format).

It confirmed the trickiest visual pieces were actually correct — a two-marker gauge
on the RDR Optimizer page that looked ambiguous in a screenshot was checked by
measuring actual DOM element positions rather than eyeballing pixels, and it landed
exactly where the underlying math said it should.

It also found one genuine, unglamorous bug that self-review had missed: on the Cost
Sensitivity dashboard, the "Analyst labor cost per case" slider was read from the
UI and then never used by anything on screen — dragging it visibly did nothing,
directly contradicting the page's own caption that promises the sliders respond
live. Fixed by making every escalated case carry a real review-labor cost that
scales with the slider, deducted from the headline "net value" number — verified
numerically (a ~₹97,000 swing moving the slider from one end to the other) before
touching the UI at all.

---

## 6. Making the prevention model genuinely better — and finding two real bugs along the way

With two days of runway left, the question became: can the prevention model
actually catch more, honestly — not by moving the threshold, not by picking a
lucky batch, but by giving it real signal it doesn't currently have?

**The gap that got found**: the model only ever saw a transaction's own
attributes (amount, category, device, day of week) — it had *no* concept of
customer identity at all. But `src/features.py` already had a proven pattern for
exactly this, built for the *other* model: a customer's prior transaction history —
computed strictly from transactions *before* the one being scored, so it's real
signal a merchant genuinely has at payment time, not a leak from the future. The
prevention model just wasn't using it.

Before touching any model code, the underlying signal was verified to actually
exist: fraud-prone customers, drawn from the same simulated population, realize a
**35.1%** dispute rate on their transactions, versus **14.9%** for everyone else —
more than double. That's real, and it's exactly the kind of thing a customer's own
history should be able to pick up on.

**This is where it got interesting.** The first retrain came back *worse*, not
better — ROC-AUC dropped to 0.497, which is statistically indistinguishable from
flipping a coin, despite the real signal that had just been confirmed to exist.
That's not a "needs more tuning" result; a result that bad, on data with genuine
signal, means something is broken, not under-optimized.

The cause, found by directly comparing feature rows against the raw data row by
row rather than trusting the number: a helper function was re-sorting data that
its caller had *already* sorted. That sounds harmless — sorting already-sorted
data should be a no-op — except the dataset's dates only have day-level precision
across 120,000 rows, meaning roughly 164 transactions share any given date on
average. Sorting the same data twice, independently, reordered those same-day ties
*differently* the second time, which silently mismatched which features belonged
to which label for most of the dataset. The fix was to stop re-sorting at all —
the function now asserts its input is already sorted instead of trying to fix it
itself — and a second instance of the exact same bug class was found and fixed in
Demo Mode's own reference-calibration code (which was sampling a subset of rows
*before* computing features, truncating each customer's visible history to
whatever happened to land in the random sample).

**The real, final result**, after both bugs were fixed and verified (18,000-row
held-out test set): PR-AUC 0.353 → **0.379**, ROC-AUC 0.658 → **0.676**, recall
21.2% → **23.1%**, precision 42.4% → **46.1%**.

That's a real improvement, not a dramatic one — and that's exactly what should be
expected. The signal in the simulated data comes from one binary "is this a
chronic-abuser customer" flag affecting about 7% of customers, not the full
spectrum of customer trustworthiness. A modest, explainable gain like this is more
trustworthy than a dramatic one would have been; a huge jump right after fixing an
alignment bug would have been the number worth being suspicious of, not this one.

---

## 7. The throughline

Almost everything in this session follows the same shape: find a real gap, verify
the underlying signal or claim is actually true *before* building on it, build the
smallest honest version of the fix, and check the result against reality rather
than trusting that the code did what it was supposed to. Several things that looked
right on first pass turned out not to be — a bar chart's math, a locale rendering
quirk, a dead slider, a feature-alignment bug that silently destroyed a model's
signal — and every one of them got caught by checking, not by assuming.

The project's actual pitch was never "this model is great." It's "here's a system
that tells you the truth about its own limits, including when that truth is
unflattering" — and the honest way to build a demo like that is to make sure the
demo is telling the truth too.

## Where things stand

- `make test`: 124 tests passing.
- Everything above is on `main` — commits `069290e`, `70ab47c`, `b722b5f`, `5295854`.
- Not yet done: a customer selector on the Prevention Score page so the new
  history-aware scoring is demoable there too, not just inside Demo Mode; and the
  pitch script (`pitch-video.md`) hasn't been updated to reference the customer-
  history work, since it postdates the last script pass.
