# HCAI design history: from prototype to the current system

This document exists to let you (and a coworker) fully understand *why* this system looks the way
it does — every major design decision, every problem that forced a redesign, and which Human-Centered
AI (HCAI) principle each decision serves — so it can be written up accurately for the paper. It also
flags, explicitly, where the current `.tex` paper draft describes something different from what's
actually implemented (Part IV) — read that section before writing the Methodology section.

Three earlier documents already exist and are the primary sources for this one:
- `docs/ai_documentation` — the original architecture rationale (predates this session).
- `docs/implementation.md` — the full-stack wiring + the Version O/N experiment design.
- `docs/architecture_change_summary.md` — this session's first four fixes, in paper-ready prose.

This document doesn't replace them — it's the connective narrative across all of it, plus the parts
that were never written up (Part III), plus the paper-vs-reality reconciliation (Part IV) nobody had
done yet.

## Executive summary

The system is a movie recommender built explicitly as an HCAI case study, not just a recommender with
an explanation bolted on. Three pillars run through every version of the design:

1. **Interpretability by construction.** The model's internal representation *is* a set of named movie
   genres, not an opaque embedding that gets explained after the fact. This is a structural choice
   (bottleneck width = number of genres), not a post-hoc interpretability technique.
2. **Human-in-the-loop control.** Users can inspect and edit the AI's inferred preferences, and — after
   this session's changes — those edits actually persist and keep shaping what they see.
3. **Transparent, faithful explanation.** The system doesn't just show *a* number; it tries to show the
   number that actually explains why *this* recommendation happened, and — after this session's
   changes — keeps that promise even as the underlying ranking mechanism changed.

The system was also built to be *empirically testable*: a between-subjects A/B experiment (Version O =
transparent/editable, Version N = black-box) with a standard SUS usability questionnaire is baked into
the application itself, not just into a study protocol layered on top.

Read Part IV before you write anything citing the recommender's architecture, the XAI method, the
dataset, or the experiment design — several claims in the current `.tex` draft don't match what's
actually running.

---

## Part I — Original design (predates this session)

### The interpretable genre bottleneck

> "The core design principle is that the user's latent profile is expressed in a **human-readable
> genre space** (18 dimensions) rather than an anonymous bottleneck, making every part of the system
> interpretable without sacrificing predictive quality." — `docs/ai_documentation`, §1

The autoencoder's bottleneck width is set *equal to the number of movie genres* on purpose. A normal
autoencoder would use some arbitrary embedding size (32, 64, 128 — whatever balances capacity against
overfitting) and the resulting latent dimensions would mean nothing to a human. Here, dimension *i* of
the bottleneck corresponds by construction to genre *i* — "Action," "Comedy," "Sci-Fi," and so on. A
user's "profile" isn't a black-box vector that needs a separate explanation model bolted onto it — the
profile *is* the explanation, already in a vocabulary a human uses naturally.

**HCAI principle: interpretability by construction, not post-hoc.** Most XAI techniques (SHAP, LIME,
saliency maps) explain a black-box model after training, approximately, and only locally. This design
instead constrains the model itself so the "explanation" is the literal internal state, exactly, not an
approximation of it. The tradeoff is capacity: an 18–19-dimensional bottleneck is a much harder
information bottleneck than a typical 64–256-dim embedding, so the model has structurally less room to
represent nuance that doesn't map onto genre — a real cost paid for the interpretability guarantee.

### Knowledge injection and the "elastic leash"

The bottleneck being genre-shaped doesn't automatically make it *stay* genre-shaped once gradient
descent starts training it — nothing stops the optimizer from drifting the encoder into an arbitrary,
uninterpretable representation that merely happens to have the right number of dimensions. The original
design addressed this in two layers:

1. **Knowledge injection at initialization** — the encoder's weights are *seeded* directly from the
   real genre-membership matrix (1 if a movie belongs to a genre, 0 otherwise), so before any training
   happens, the bottleneck already means exactly what its dimensions claim to mean.
2. **"Elastic leash" regularization** — a soft loss term (`L_semantic`, MSE between the current encoder
   weights and the genre-prior matrix) plus a *hard* constraint (weight clipping to within `ε = 0.15` of
   the prior after every optimizer step) keep training from drifting the encoder away from that
   semantic anchor, while still allowing enough movement (`ε`, not zero) for the model to actually learn
   from data.

**HCAI principle: faithfulness/semantic grounding.** An interpretable-by-construction bottleneck is
only trustworthy if training doesn't quietly erode the construction. The "leash" is a direct, quantified
tradeoff between two things a human-centered system needs simultaneously: it must *learn* (predict
ratings well) and it must *stay honest* (the "Action" dimension must still mean Action after training).
`ε` is the literal dial on that tradeoff — tighter leash, more faithfulness, less learning capacity;
looser leash, the reverse. **This is the exact tension Part II §1 later revisits** — the leash mechanism
turned out to have a real flaw (documented there) that this design couldn't see from first principles
alone, only from testing at scale.

### The Human Override System

`docs/ai_documentation` §4 names this explicitly as an HCAI feature, with three levels of increasing
sophistication:

- **Level 1 — Direct profile editing.** The user sets genre values by hand; the decoder translates that
  directly into movie scores with no AI correction layered on top. The model is used purely as a
  "genre-to-film translator" at this point — full human control, zero AI opinion.
- **Level 2 — Override-impact transparency.** After an override, the system shows *which specific
  movies* were most boosted or suppressed by the change — not just a new ranked list, but the *delta*
  the human's edit actually caused.
- **Level 3 — Hybrid override path.** The user expresses preference changes in natural-language
  categories (`strongly_increase`, `slightly_reduce`, etc.), mapped to floats and blended *additively*
  with the AI's own output via a strength parameter `α`, rather than fully replacing the AI's judgment.

**HCAI principle: human-in-the-loop control.** These three levels form a genuine spectrum from "AI
suggests, human decides everything" (Level 1) to "AI and human blend" (Level 3) — deliberately not a
single fixed answer to "how much control should the user have." The Level 3 blending design in
particular already anticipated something important: giving a user a slider that can swing a genre from
0% to 100% risks letting one bad-mood click nuke every future recommendation. Blending with `α` was the
original answer to that risk. (This session's redesign reaches a related but different answer — see
Part II §3 — and is worth explicitly comparing to this original approach in the paper.)

### XAI methods

Four explanation mechanisms were originally specified, each trading off differently between fidelity,
speed, and scope:

| Method | Type | Scope | Tradeoff |
|---|---|---|---|
| Soft rationale (fuzzy thresholding) | Local, heuristic | Per movie | Fast, human-readable sentence; not a formal attribution method |
| SHAP (KernelSHAP, decoder only) | Local, exact | Per movie, per genre | Principled game-theoretic attribution; applied only to the *decoder* sub-network specifically **because** the genre bottleneck is already interpretable — SHAP only needs to explain the last, small step (genre → movie score), not the full ~9,742-dimensional input space |
| LIME (Gaussian perturbation + Ridge, genre space) | Local, approximate | Per movie, per genre | Fast, intuitive; documented limitation: "results vary across random seeds," i.e. not reproducible run-to-run — the doc itself flags SHAP as the more trustworthy choice for anything quantitative |
| Permutation feature importance | Global, model-agnostic | Whole model | Answers "which genre matters most across all users," not "why did *I* get *this* recommendation" |

**HCAI principle: multi-modal explainability, matched to the question being asked.** No single
explanation method answers every "why" a user might have — a fast heuristic sentence for a quick glance,
an exact local attribution for a rigorous per-decision question, and a global importance ranking for
"what does this system care about overall" are three different questions. The explicit SHAP-on-decoder
design choice is itself a direct product of the interpretable-bottleneck decision (Part I §1) — it's
cheap and exact *because* the space it has to explain is small and already meaningful, which wouldn't be
true if the bottleneck were an arbitrary embedding.

### The streaming-pipeline rewrite

The original prototype trained and served on **MovieLens ml-latest-small** — 100,836 ratings, 9,742
movies, 610 users, 18 genres — small enough that a dense `(610 × 9,742)` ratings matrix fits trivially
in RAM. Scaling to the real, full **ml-latest** dataset (~33.8M ratings, ~86,537 movies, ~330,975 users)
broke that assumption outright: a dense matrix at that scale would need **~115 GB of RAM**
(330,975 × 86,537 × 4 bytes), far beyond any reasonable machine.

The rewrite (commit `59308c3`, "Add streaming ML pipeline, HCAI model, XAI") replaced the monolithic
`ai.py` with five focused modules (`data_pipeline.py`, `id_mapping.py`, `model.py`, `losses.py`,
`xai.py`) built around streaming `ratings.csv` line-by-line through a PyTorch `IterableDataset`, capping
peak RAM at `O(batch_size × num_movies)` instead of `O(num_users × num_movies)`. The interpretable
genre-bottleneck design itself (Part I §1) carried over unchanged — this was a scaling/engineering
rewrite, not an architecture rethink; genre count grew from 18 to 19 simply because the larger, real
dataset's actual genre vocabulary (discovered dynamically from `movies.csv`, not hardcoded) has one more
entry than the small prototype's.

**HCAI principle: usability requires the system to actually run at real scale.** An interpretable,
faithful, human-controllable model that can only ever run on a 610-user toy dataset isn't actually
human-centered in any deployable sense — "human-centered" has to include the humans who'd want to use
the *real* system, not just the researchers testing a prototype. This is easy to overlook when framing
HCAI purely as an ML-architecture property; here it was a hard infrastructure constraint that had to be
solved before any of the interpretability/control work could reach real users.

### The Version O/N A/B experiment design

This is the actual research instrument the paper is built around, and it's implemented in the
application itself, not just described in a protocol document:

- At registration, `auth_service.register()` calls `random.choice(["O", "N"])` once and stores the
  result permanently on the user's account (`users.version` column) — **a between-subjects design**:
  each participant experiences exactly one condition, forever, for that account.
- **Version O ("Transparent AI")** shows the "Edit Preferences" button on every recommendation card,
  loads XAI explanations on expand, and gates "Continue to Survey" behind `has_edited = 1` (the user
  must apply at least one edit before proceeding) — enforced in both the UI and the database.
- **Version N ("Standard AI")** hides all of that; the user sees a plain ranked list and can proceed to
  the survey immediately.
- Both conditions answer the identical 10-question Brooke (1996) System Usability Scale (SUS)
  questionnaire plus three demographic questions (age group, degree/job, streaming-platform experience)
  afterward.

> "The research hypothesis is that users exposed to transparent, editable AI recommendations (Version
> O) will report higher system usability (SUS score) and better understanding of recommendations than
> users in the black-box condition (Version N)." — `docs/implementation.md`, §3

**HCAI principle: empirically testing transparency's effect on trust/usability, not just asserting it.**
Everything in Parts I through III of this document is a design choice justified by an HCAI *principle*
in the abstract — this section is the mechanism that turns "we believe transparency helps" into a
falsifiable, measured claim. It's the actual contribution the paper is meant to report on.

---

## Part II — This session's four core iterations

`docs/architecture_change_summary.md` documents these in full technical detail with citable
"Framing for the paper" language; this section gives the shorter narrative arc and the explicit HCAI
framing for each.

### 1. Genre-bottleneck calibration

**Problem.** Real users reported genre-affinity percentages clustering almost entirely near 0% or 100%,
with almost nothing in between — an "interpretable" profile that in practice just says "yes" or "no" per
genre, discarding all the graded nuance interpretability was supposed to preserve.

**Diagnosis.** Two independent causes, found by loading the actual trained checkpoint and simulating
synthetic users against its real weights: (a) the anchored genre logit was a raw **sum** of a user's
ratings in that genre, not an average — so a user with 1 rating and a user with 200 ratings in the same
genre produced wildly different-scale logits that no single global temperature scalar could reconcile;
(b) roughly 85% of the hidden layer fed the same bottleneck completely unregularized — the "elastic
leash" (Part I §2) only ever constrained the prior-anchored slice of the weights, never this much larger
"free" pathway, which was independently contributing uncontrolled swings of ±4 to ±106 to the logit.

**Fix.** Replaced the learned, weight-clipped sum with a **closed-form, Bayesian-smoothed, count-
normalized average** computed directly from the genre-membership matrix at every forward pass (exact,
not approximated by a drifting weight matrix — so the elastic leash and its `ε` tradeoff, Part I §2,
became unnecessary for this component entirely), plus a small `tanh`-bounded residual pathway
(replacing the unregularized free pathway) for genuine nuance the closed-form average can't capture.

**HCAI principle: meaningfulness of the interpretable representation.** Interpretability-by-construction
(Part I §1) guarantees you can *point at* what a dimension means. It does not automatically guarantee
the *values* in that dimension carry real information — a genre score that's always ~0% or ~100%
regardless of actual preference strength is technically "interpretable" (you know what it represents)
while being nearly useless (it tells you almost nothing graded). This iteration is the difference
between interpretability as a labeling property and interpretability as an *informativeness* property —
a distinction worth making explicit in the paper, since it's easy to conflate the two.

### 2. Recommendation ranking: personalization lift and a quality floor

**Problem.** Certain movies — documentaries were the clearest case — kept appearing in recommendations
regardless of a user's actual profile.

**Diagnosis.** The decoder has a per-movie bias term, learned independently of any genre signal,
representing a movie's *generic* appeal. Confirmed by running the decoder on a fully genre-neutral input
(0.5 for every genre, the "no information at all" point) and inspecting which movies still scored
highest — several documentaries ranked in the global top-20 purely on generic bias, alongside genuinely
broadly-loved films the model can't distinguish from them by output alone (both present as "high score
regardless of profile," for different real-world reasons: documentaries attract a small, self-selecting,
enthusiastic rater population with few negative ratings, while broadly-loved films are genuinely broadly
loved).

**First attempt, rejected.** Rank purely by "lift" (predicted score minus the genre-neutral baseline) to
isolate the personalized signal. This has a real failure mode, caught by testing before shipping: a
genuinely bad movie with an equally low baseline can show *positive* lift merely by being "less bad than
expected," which surfaced near-unwatchable films (e.g. *Birdemic: Shock and Terror*) ahead of much better
predictions. Lift alone conflates "unusually good fit for you" with "merely less-bad-than-expected."

**Final design.** Only rank by lift among candidates whose *raw* predicted score already clears a
quality floor (3.0/5); everything below the floor is ranked by raw score at the bottom, never promoted
by lift. This is what later motivated **Part III §1**: rather than picking one ranking philosophy and
hiding the other, the UI now shows both lists side by side.

**HCAI principle: personalization vs. popularity bias.** This is a direct instance of a well-known
recommender-systems fairness concern (popularity bias / filter-bubble-adjacent effects) — a system that
just serves generically popular content isn't really "recommending," it's aggregating. The failed first
attempt is worth including in the paper as a cautionary methodological note: a well-motivated fairness
intervention (personalize, don't just rank by raw appeal) can itself introduce a *different* failure mode
if applied without a floor — bias-correction techniques need their own validation, not just a plausible
rationale.

### 3. Persisted profile overrides

**Problem.** Editing genre preferences only ever affected the single response it was submitted with —
revisiting the profile or recommendations page reverted to the pure AI-inferred state, silently
discarding the user's correction. The Human Override System (Part I §3) existed in the UI but had no
lasting effect.

**Fix.** Edits are now persisted server-side as **deltas** (e.g. "+0.25 to Action"), not absolute values,
in a new database table, merged onto the AI-inferred profile on every read (`GET /api/profile`,
`POST /api/recommend`, `POST /api/explain`) until the user changes or explicitly resets them. Storing a
*delta* rather than an absolute value keeps the edit meaningful even as the AI's own estimate continues
to shift with new ratings — "boost this a bit more than baseline" stays interpretable regardless of what
the baseline currently is, where a stored absolute value would silently go stale.

Fixed alongside this: the interactive-override mechanism (`forward_interactive`) had been called with a
hand-built vector representing the *entire* genre bottleneck, so any genre the UI hadn't explicitly
included in a request was implicitly zeroed rather than left at its AI-inferred value — a real bug,
not just an incompleteness.

**HCAI principle: control that persists is the only control that's real.** A feature that lets you
"correct" the AI but forgets the correction the moment you navigate away isn't meaningfully offering
control — it's a demo of control. Notably, **this design change actually validates the original paper
draft's own stated intent** (`.tex` §3: edits should "nudge," not fully override, preference scores, to
prevent one bad-mood click from zeroing out a criterion) — the delta-based mechanism achieves exactly
that nudge property, just via a different implementation path (persisted per-genre deltas, merged onto
the AI's own inference) than the paper's originally-described per-movie five-point scale (see the
discrepancy noted in Part IV). This is a good example of two independently-arrived-at designs converging
on the same underlying HCAI principle from different directions.

### 4. Lift-aware XAI rationale

**Problem.** Once ranking changed to lift-based (Part II §2), the natural-language "why was this
recommended" explanation didn't change with it. It could still only explain recommendations via
genre-dominance ("your Action profile is high..."); for anything ranked by lift instead — including,
concretely, movies like *Shawshank Redemption* — it fell back to a generic, uninformative message
("recommended based on general collaborative filtering patterns") that described neither the real reason
the movie ranked nor anything specific to the user.

**Fix.** The rationale generator now takes the same `predicted_score` and `baseline_score` the ranking
mechanism itself uses, and names whichever of two real mechanisms actually applies: high lift ("fits your
specific taste beyond generic appeal") or low lift with a high absolute score ("broadly well-liked, and
your own ratings don't contradict that").

**HCAI principle: faithfulness of explanations.** This is a standard XAI evaluation criterion — an
explanation should accurately describe the model's *actual* decision process, not just sound plausible.
The generic fallback here wasn't *wrong* exactly, but it was **unfaithful**: it implied a mechanism
(collaborative-filtering pattern-matching) that wasn't what actually drove that specific ranking. A paper
built around measuring "trust from transparency" should treat this as a first-class risk, not a cosmetic
bug — an explanation that quietly stops matching reality as the underlying system evolves is arguably
*worse* for trust than no explanation, because a user has no way to know the explanation has become
unreliable.

---

## Part III — Final UX/interaction round

This round hasn't been written up anywhere else yet; the following is sourced directly from this
session's implementation work.

### Two recommendation lists: "Top Rated" and "For You"

Directly downstream of Part II §2's ranking redesign: rather than picking one ranking philosophy (raw
score vs. lift) and hiding the other, the Recommend page now shows both, computed from the same
underlying predictions (no extra model cost — just two different sort orders over one forward pass),
each with a one-line plain-language explanation of what it means (*"Movies we predict you'd rate the
highest, period — including broadly popular titles most people enjoy"* vs. *"Matched to your specific
taste, not just what's generally popular — these may score a bit lower but fit you better"*).

**HCAI principle: transparency about trade-offs, not hidden curation.** Any single ranked list is an
editorial choice the system makes silently on the user's behalf — "raw score" and "lift" are both
defensible answers to "what should I recommend," and picking one without saying so hides that a choice
was even made. Surfacing both, labeled, turns an invisible design decision into something the user can
see and reason about themselves.

### All-genre explanations

The Profile page's per-genre "Why?" explanation previously only covered the top 5 genres by score (both
for cost reasons and because that's what an earlier iteration defaulted to). It now covers all 19
genres, still fetched lazily on first click so the cost is only paid if a user actually asks.

**HCAI principle: completeness of explanation access.** Limiting explanations to only the "important"
(highest-scoring) genres implicitly tells the user which questions are worth asking — but a user might
specifically want to know *why their Documentary score is low*, which a top-5-only design would never
let them ask. Completeness of *access* to explanation (even if most users never use it) is a different
property from the *content* of any one explanation (Part II §4), and both matter for a transparency
claim to be honest.

### Refresh Suggestions and half-star ratings

Two smaller frictions, both about matching the interface to real human judgment rather than an
implementation convenience: (a) the initial movie-rating batch was a fixed 50 titles — if a user
recognized few of them, they had no way to get more options without the AI profile being built from too
few ratings; a "Refresh Suggestions" action now swaps in a new random batch, excluding what's already
been shown, while keeping any ratings already given. (b) Star ratings were locked to whole integers even
though the underlying MovieLens training data (and the model) support 0.5-star granularity — a user
whose true opinion is "3.5 stars" was previously forced to round, silently injecting noise into their own
profile before the model ever saw it.

**HCAI principle: reducing friction / matching real human judgment granularity.** Neither of these
changes the model. Both change how faithfully the *interface* can capture what a human actually thinks —
a system can be maximally interpretable and controllable in principle and still fail a user if the input
affordances themselves are coarser than the preferences they're meant to capture.

### Reset to AI Profile

A user can now explicitly discard all saved overrides (Part II §3) and revert to the pure AI-inferred
profile in one action.

**HCAI principle: reversibility of override state.** Persisted control (Part II §3) that can't be undone
is a different kind of trap than ephemeral control that doesn't stick — a user should be able to try an
edit, decide they preferred the AI's original judgment, and get back to it without re-registering or
manually undoing each change one at a time.

### Dev-only A/B preview toggle

A developer/researcher-only toggle lets someone preview both Version O and Version N's UI without
re-registering a new account, purely for local testing — it never touches the real, permanently-assigned
`user.version` a study participant would experience.

**This is a research-methodology tooling note, not an HCAI design decision** — flagging it as
categorically distinct from the rest of this section so it doesn't get cited as a user-facing feature by
mistake.

---

## Part IV — Where the paper draft conflicts with the real system

The current `docs/AI_User_Profiling.tex` draft is mostly ACM template boilerplate (empty Abstract, empty
Research Questions/Hypotheses, empty Results, empty Conclusion, placeholder Lorem-ipsum appendix). The
real content that does exist — §3 "Recommender Design" and §4 "Experiment" — describes a system that
differs from what's actually implemented in several concrete ways. **Flagged here for you and your
coworker to reconcile deliberately; the `.tex` file itself was not touched.**

| Claim in the `.tex` draft | What's actually implemented | Source |
|---|---|---|
| "The recommender-System is an Convolutional Neural Network (CNN)" | `DualModeHCAIAutoEncoder` — a genre-bottleneck autoencoder (Linear → ReLU → Linear encoder, mirrored decoder). No convolutional layer anywhere in the model. | `backend/app/ai/model.py` |
| "the MovieLens 32M dataset... 30 million movie ratings, 87,585 movies and 200,948 users" | The actual dataset in use is MovieLens **ml-latest** — 33.8M ratings, 86,537 movies, 330,975 users. This is a *different* official MovieLens release from "32M," and the counts given don't match either release exactly. | `backend/app/config/config.py`, `backend/app/ai/data_pipeline.py`, `SETUP.md` |
| "Transparency is created with SHAP and LIME" | The live `xai.py` module implements a custom natural-language soft-rationale generator plus leave-one-out permutation importance — no SHAP or LIME call in the current, running code path. `shap`/`scikit-learn` imports exist only in the dead, never-imported legacy `backend/app/ai/ai.py` (confirmed by this session's `requirements.txt` cleanup, which removed them as unused). SHAP/LIME *were* real, implemented methods in the original prototype (`docs/ai_documentation` §5) — they did not survive into the streaming-era rewrite. | `backend/app/ai/xai.py`, `docs/ai_documentation` §5 |
| "the K most important criteria in regards to the movie suggestion... on a five point scale" (per-movie editing) | Editing is **per-genre and global**, not per-movie: boost/suppress buttons on a genre apply to the user's whole profile via 5 discrete levels (−−/−/○/+/++), not to "the K most important criteria" of one specific movie. | `frontend/src/pages/ProfilePage.jsx`, `frontend/src/pages/RecommendPage.jsx` |
| Within-subjects, counterbalanced design (Group A: O then N; Group B: N then O), each participant experiences both conditions | **Between-subjects**: `auth_service.register()` calls `random.choice(["O", "N"])` exactly once at registration; the assignment is permanent for that account. A participant only ever experiences one condition. | `backend/app/services/auth_service.py`, `docs/implementation.md` §3 |
| "rate random movies with either like, dislike or undecided... until 10 movies got either the rating like or dislike" | Actual rating UI is a 0.5–5.0 star scale (not like/dislike/undecided), from a fixed initial batch of 50 movies (now with a "Refresh Suggestions" option) — there's no code-enforced "must rate exactly 10" gate before proceeding. | `frontend/src/pages/RatingsPage.jsx`, `frontend/src/components/StarRating.jsx` |
| Title: "...an Edibility and Transparent Recommender..." | Almost certainly intended as "**Editability**" — noted here as the paper's own chosen (non-standard) term, not something this document is correcting on your behalf. | `docs/AI_User_Profiling.tex` |

None of this means the paper draft is *wrong to have existed* — it reads like an earlier design intent
(possibly written before, or independent of, the implementation that was ultimately built) that the
implementation diverged from as real constraints (dataset scale, library availability, testing results)
were discovered. The important thing for the paper is to pick, deliberately, which of these descriptions
is the one being reported on, and make the write-up and the code agree.

---

## Part V — Open gaps without documented rationale

Things that are true of the current system but don't have a stated "why" anywhere, worth flagging as
honest limitations rather than treating as principled choices if the paper discusses them:

- **Hyperparameters** (`hidden_dim=128`, `lr=0.01`, `epochs`, `dropout=0.4`, and this session's additions
  `count_smoothing=1.5`, `rating_midpoint=2.5`, `residual_scale` init `0.5` clamped to `[0, 2.0]`,
  quality-floor `3.0`) are all stated as fixed values but only justified as "this empirically fixed the
  observed problem," not derived from a stated principle or swept systematically. Fine to report as
  empirically-tuned, but shouldn't be described as principled/optimal without further work.
- **The root `README.md` is stale and describes a system that doesn't exist** — it documents a
  MongoDB + LangChain + OpenAI/Anthropic + FAISS RAG architecture that was apparently an earlier plan
  and never actually built; the real system uses SQLite and a from-scratch PyTorch autoencoder, no LLM
  or vector database anywhere. Don't treat `README.md` as ground truth for the paper — `SETUP.md` and
  this document are current.
- **`docs/implementation.md`'s own architecture diagram is internally inconsistent** — it still labels
  `app/ai/ai.py` as "unchanged" in one place while §4 of the *same document* describes it being replaced
  by the five-module streaming rewrite. Use §4's text, not the top-of-document diagram.

---

## Quick-reference table

| Design decision | HCAI principle(s) | One-line tradeoff |
|---|---|---|
| Genre-sized bottleneck | Interpretability by construction | Less latent capacity than an arbitrary embedding |
| Knowledge injection + elastic leash | Faithfulness / semantic grounding | Rigidity (`ε`) vs. learnability |
| Human Override System (3 levels) | Human-in-the-loop control | More control surface = more ways to misuse it (addressed by nudge/blend design) |
| Multi-method XAI (soft/SHAP/LIME/permutation) | Multi-modal explainability | Fidelity vs. speed vs. reproducibility, per method |
| Streaming pipeline rewrite | Usability at real scale | Engineering complexity for deployability |
| Version O/N A/B experiment | Empirical validation of transparency's effect | Between-subjects costs statistical power vs. within-subjects, but avoids order/carryover effects |
| Count-normalized genre calibration | Meaningfulness of the interpretable representation | Closed-form exactness vs. flexibility of a learned weight matrix |
| Lift ranking + quality floor | Personalization vs. popularity bias | "For You" scores may look lower than "Top Rated" even when better matched |
| Persisted delta-based overrides | Control that persists is real control | Must handle staleness/merge semantics as the AI profile itself keeps changing |
| Lift-aware XAI rationale | Faithfulness of explanations | Explanation logic must be kept in sync with ranking logic going forward, or it degrades again |
| Two recommendation lists | Transparency about trade-offs | More UI surface / user has to interpret two lists instead of one |
| All-genre explanations | Completeness of explanation access | Slightly higher compute cost per genre explained (mitigated by lazy fetch) |
| Refresh Suggestions, half-star ratings | Reducing friction / matching human judgment granularity | Small UI complexity increase for meaningfully less input noise |
| Reset to AI Profile | Reversibility of override state | None significant — pure usability win |

---

## Part VI — Core concepts explained (for your own understanding, not paper text)

Everything above this line is written to be quoted or adapted for the paper. Everything below is not —
it's a plain-language walkthrough of the newer mechanisms (Part II), for actually understanding what the
code does, with worked numbers pulled from real testing this session. Skip this if the concepts already
make sense from the sections above; read it if a phrase like "count-normalized Bayesian-smoothed average"
reads as jargon rather than as something you could explain to someone else.

### 1. The count-normalized, Bayesian-smoothed genre average

**The old approach, and why it broke.** The original design computed a genre score by *adding up* a
user's ratings in that genre. The problem: the total just keeps growing the more a user rates. Someone
with 1 rating and someone with 200 ratings in "Action" end up with totally different-sized numbers, even
if both of them equally love Action — there's no single conversion factor that turns both of those
into a sensible, comparable 0–100% score.

**The fix, in plain language.** Instead of a *total*, compute an *average* — but a cautious one. If a
user has only rated one movie in a genre, that single data point shouldn't be treated as fully reliable
evidence; the system partially discounts it toward "neutral" (2.5 stars, the exact middle of the 0–5
scale) until more ratings accumulate to back it up. This is a standard statistical technique called
Bayesian (or Laplace) smoothing — you can think of it as mixing in a fixed number of imaginary "neutral"
ratings alongside the real ones, so a genre with little real evidence stays close to neutral, and a genre
with lots of real evidence is barely affected by the imaginary ones.

**The formula, worked through with real numbers** (`count_smoothing = 1.5`, i.e. "1.5 imaginary neutral
ratings" mixed in):

```
genre_mean = (sum of the user's ratings in this genre  +  1.5 × 2.5)
             ─────────────────────────────────────────────────────
             (number of ratings in this genre  +  1.5)
```

| Situation | Calculation | Result | Interpretation |
|---|---|---|---|
| 0 ratings in "War" | (0 + 3.75) / (0 + 1.5) | **2.5** | Exactly neutral — no evidence at all, so no opinion |
| 1 rating of 5★ in "War" | (5 + 3.75) / (1 + 1.5) | **3.5** | Nudged up, but nowhere near 5 — one data point isn't fully convincing |
| 10 ratings averaging 4★ in "Action" | (40 + 3.75) / (10 + 1.5) | **3.80** | Close to the true 4.0 average — 10 data points *are* fairly convincing |
| 200 ratings averaging 3.5★ | (700 + 3.75) / (200 + 1.5) | **3.49** | Essentially the true average — smoothing barely matters with that much evidence |

This `genre_mean` is then centered (subtract 2.5, so "exactly neutral" becomes 0) and divided by a
learnable "temperature" number before going through a sigmoid squashing function to produce the final
0–100% shown in the UI. Centering at 2.5 is what makes "no opinion" map to 50% rather than some other
arbitrary number — 50% is the natural midpoint for "we genuinely don't know."

**Why it matters.** This average is *count-invariant by construction* — the 1-rating user and the
200-rating user both get numbers on the same, comparable 0–5 scale before the sigmoid ever sees them.
That's the actual fix for the old 0%/10000% bug: it wasn't a display formatting issue at its core, it was
that the underlying number being displayed was never on a consistent scale to begin with.

### 2. The bounded "residual" pathway

The averaging above is deliberately simple, and simple has a real limitation: it can't capture
*interaction effects* — e.g. maybe someone who rates both Horror and Comedy highly specifically loves
horror-comedies, a pattern a per-genre average can't see because it looks at each genre independently.

To allow for this without reintroducing the old problem, there's a second, small pathway (a couple of
neural-network layers) that's allowed to add a bit of extra adjustment on top of the honest average — but
it's deliberately kept weak. Its output is squashed through a `tanh` function (which mathematically can
never produce a number outside roughly −1 to +1) and then multiplied by a small, capped scaling factor
(at most 2.0), so no matter how the network's internal weights end up, this pathway can only ever nudge
the final score by a bounded amount — it can add flavor, but it can't take over.

**Why the cap specifically matters here:** the *old* design had an equivalent "extra" pathway with no
limits on it at all, and when the trained model's weights were actually inspected, that unconstrained
pathway alone was found to be capable of swinging a score by as much as ±106 — on a scale where the
entire useful range is roughly 0–5. That's not a minor wobble, that's enough to single-handedly cause the
exact same saturation problem all over again, just through a different part of the network. Bounding this
pathway is a direct, deliberate fix for that specific, measured failure.

### 3. "Lift" and the genre-neutral baseline

Every movie in the catalog has some amount of *generic* appeal — how good the model thinks it is for a
hypothetical person who has given zero information about their taste. You can measure this directly: feed
the model a perfectly neutral input (every genre at exactly 50%, i.e. "I haven't told you anything") and
see what it predicts. Whatever score a movie gets under that condition is its **baseline**.

Concretely, from real testing this session: *Planet Earth II* had a baseline around 4.45/5, and *Band of
Brothers* around 4.48/5 — both very high, **regardless of who's asking**. That's exactly the problem: a
movie like that would show up near the top of *everyone's* recommendations whether or not it actually
matches their taste, simply because the model has learned it's broadly appealing.

**Lift** is defined as:

```
lift = (this specific user's predicted score)  −  (that movie's generic baseline)
```

A movie with high lift is one the model thinks is meaningfully *better for this particular user* than for
an average stranger — i.e. something that's actually personalized, not just something that would appear
on any list regardless of who's looking at it.

### 4. Why lift alone wasn't safe: the quality floor

Lift on its own has a blind spot: it only measures *relative* improvement, never whether the movie is
actually any good in absolute terms. A movie that's bad for basically everyone has a *low* baseline
(nobody expects to like it) — and if one particular user's predicted score for it happens to be even
slightly less catastrophic than that already-low baseline, the lift number comes out **positive**, even
though the movie is still bad.

This wasn't theoretical — it was caught directly by testing: turning on pure, unconditional lift-ranking
put *Birdemic: Shock and Terror* (a notoriously, almost unwatchably bad cult film) and several similarly
poor movies at the **top** of the recommendation list, ahead of movies that were actually predicted to be
good. The lift calculation was technically working exactly as designed — it just turned out "designed
correctly" wasn't the same as "safe to use unconditionally."

**The fix:** lift is only allowed to influence the ranking among movies that already clear a minimum
absolute quality bar — predicted 3.0 stars or better, on the 0–5 scale. Below that bar, movies rank by
their plain predicted score instead, so genuinely bad movies stay at the bottom no matter how their lift
number looks. Above the bar — among movies already predicted to be good — lift decides the order, so
options that are unusually well-matched to *this specific user* get pushed ahead of things that are
merely broadly popular.

### 5. Delta-based persisted overrides

When a user clicks "boost Comedy," the system doesn't record "Comedy = 80%" as a fixed number. It records
a **relative adjustment** — "Comedy: +0.25" (on the model's internal 0–1 scale) — meaning "whatever the
AI currently believes about Comedy, nudge it up a bit from there." This adjustment is saved permanently
in the database and gets **re-applied fresh** every single time the AI computes a new opinion about that
user.

Why a relative delta instead of an absolute value: the AI's own opinion about a user keeps changing as
they rate more movies. A saved absolute value ("Comedy = 80%") would silently go stale — it would stop
meaning "boosted relative to what the AI thinks" and start meaning "hardcoded to 80% forever, regardless
of anything new the AI learns." A saved delta stays meaningful indefinitely, because it's always
interpreted relative to whatever the AI's *current* estimate is, not a snapshot frozen at the moment the
user clicked the button.

### 6. The two-branch, lift-aware explanation

Since a movie can now be recommended for two genuinely different reasons — it strongly matches a genre
the user clearly loves, *or* it's just unusually well-suited to them personally even without one standout
genre — the explanation text checks which situation actually applies and describes that one:

- **If a genre clearly dominates** (the user's affinity for a genre the movie belongs to is well above
  their own average across all genres): *"Recommended because your Action, Adventure profile is high,
  heavily influenced by your 5.0-star rating of [movie]."*
- **If no genre dominates, but the lift is high** (concept 3 above): *"predicted 4.1/5 for you
  specifically — 0.4 stars above what a typical viewer would get — even though no single genre in your
  profile stands out as the reason."*
- **If no genre dominates and the lift is low, but the absolute score is still good:** *"widely well-liked
  regardless of genre profile, and your own ratings don't suggest you'd feel differently."*

The point of having three branches instead of one generic sentence: whichever mechanism actually produced
the recommendation, the explanation names *that* mechanism — it doesn't fall back to a vague, technically-
safe-sounding sentence that happens to not be wrong, but also doesn't say anything true and specific about
*this* recommendation.
