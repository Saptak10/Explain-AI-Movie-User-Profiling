# Writing guide: closing the gap between the paper and the real system

For your coworker, to use while writing/finishing the paper. This is deliberately organized around
**what a grader is actually checking for** in an HCAI course paper, not just "here's what the system
does" — the goal is a paper that reads as internally consistent, technically accurate, and genuinely
reflective, because those are the three things that separate a good HCAI paper from a mediocre one.

Companion document: `docs/hcai_design_narrative.md` has the full technical depth (every design
decision, every iteration, mapped to an HCAI principle). This document is the *action list* —
what to fix, what to write, and why it matters for the grade.

## TL;DR — three things to resolve before writing anything else

1. **The within-subjects vs. between-subjects mismatch is the single highest-risk issue in the paper.**
   §4 describes a counterbalanced within-subjects design (each participant does both T and C). The code
   assigns a version once, permanently, per account (`random.choice(["O","N"])` at registration — see
   Part IV below). If the actual study was run with two accounts per participant, say so explicitly and
   describe *how* the two sessions were linked for analysis. If it was actually run between-subjects
   (one account, one condition, per participant), the Experiment section needs rewriting — and so does
   whatever statistical test is planned for Results (paired vs. independent-samples). **A grader who
   works through the methodology and finds this inconsistent will flag it immediately** — it's the kind
   of thing that undermines confidence in the whole Results section even if the numbers themselves are
   fine.
2. **Three technical claims are verifiably wrong and easy to fix.** "Convolutional Neural Network,"
   "MovieLens 32M," and "bypassing the encoder" all describe something other than what's actually
   running. These are the kind of errors a technically literate grader checks first, and they're the
   cheapest possible thing to fix — the correct replacement text is below, ready to paste in.
3. **The paper has an open placeholder** — *"how are the values for the preference scores
   calculated? "* — that's now answerable precisely from the code. Answer below.

---

## What an HCAI course paper is actually graded on

Worth naming explicitly so the rest of this document makes sense as advice, not just a list of fixes.
Typically:

- **Technical accuracy** — does the described system match what was built and tested? (Currently: no,
  in three specific places — see above.)
- **Methodological rigor / internal consistency** — do the design, the data collection, and the
  statistics all agree with each other? (Currently: the within/between-subjects question is unresolved.)
- **Explicit human-centered framing** — not "we built an editable, transparent system" but *why* each
  design choice serves a specific HCAI principle, and what it cost to get there. This is the part most
  student papers do weakly (they describe *what*, not *why* and *at what tradeoff*) — and it's the part
  this project has unusually strong material for, because the system went through several real,
  documented failures and fixes. Use that.
- **Honest limitations and iteration** — a paper that only reports what worked reads as either
  incomplete or lucky. A paper that reports "we tried X, it had this specific failure mode, here's what
  we learned and changed" reads as rigorous. You have at least two genuinely good stories like this
  (below) — use them in Discussion, not just as a footnote.
- **Grounded literature connection** — the Introduction already cites the right things (He et al. for
  NCF, Tintarev & Masthoff for explanation/trust, Ribeiro et al./Lundberg & Lee for LIME/SHAP, Wu et al.
  for GNNs, Zhang & Chen for the explainability→trust survey claim). The one thing to reconcile: SHAP and
  LIME are cited as the paper's XAI technique, but the actual system doesn't use either (see below) — the
  Introduction's framing needs a one-sentence adjustment, not a rewrite.

---

## Section-by-section

### Abstract

Still template placeholder text. Write this **last**, after Results — but structurally it should state:
the system (interpretable genre-bottleneck recommender with editable preferences), the research question
(does transparency/editability improve satisfaction and trust), the method (between- or within-subjects
study, N participants, SUS-based evaluation), and a one-sentence result once you have one.

### Introduction

Mostly solid — the literature grounding is real and relevant. One adjustment: it currently sets up SHAP
and LIME (Ribeiro et al. 2016; Lundberg and Lee 2017) as the explanation techniques this project uses,
but §3 has already moved to permutation importance, and the actual code confirms that (see Part IV). Two
honest ways to handle this, either is fine:
- Keep SHAP/LIME in the Introduction as *background on the XAI landscape* (they're legitimately relevant
  citations for framing the research area), then explicitly note in §3 that this project's interpretable
  bottleneck design (see next section) meant a simpler, cheaper method was sufficient — which is actually
  a **more interesting methodological point** than just using SHAP: *"because the model's internal
  representation is already human-readable by construction, exact model-agnostic attribution (SHAP) was
  unnecessary — the genre bottleneck's own activations, combined with permutation importance over the
  user's own rating history, already provide a faithful, real-time explanation."* That's a real
  contribution to note, not a downgrade.
- Or just narrow the Introduction's XAI framing to permutation importance + soft rationale from the
  start, and cite Ribeiro/Lundberg as prior work you *considered and moved past*, not what you built.

### Section 3 — Recommender Design

Three sentences need replacing:

| Current text | Replace with |
|---|---|
| "The recommender-system is an Convolutional Neural Network (CNN)." | "The recommender is a genre-bottleneck autoencoder: a `movies → 128-unit hidden layer → 19-genre bottleneck → 128-unit hidden layer → movies` architecture, where the bottleneck's width is deliberately set equal to the number of movie genres so each latent dimension corresponds to a named, human-readable genre rather than an opaque embedding." |
| "the prominent dataset MovieLens 32M was used. The dataset contains 32 million ratings, 200,000 tags, 87,000 movies and 200,000 users" | "the MovieLens *ml-latest* dataset was used: 33.8 million ratings, 86,537 movies, and 330,975 users." (These are the exact figures from the dataset actually loaded by the code — verified directly against `movies.csv`/`ratings.csv` this session.) |
| "Edibility is achieved by bypassing the encoder and instead accepting a vector that can be edited by the user." | "Editability is achieved by letting the user apply a bounded adjustment (a discrete five-level boost/suppress choice per genre) on top of the model's own inferred genre profile, rather than replacing it outright — the encoder's inference is preserved for every genre the user hasn't touched, and only the adjusted genres are shifted." (This is the corrected, current mechanism — see Part IV's note on why the original "bypass the encoder" description is now outdated, not just imprecise.) |

The permutation-importance sentence and the "[how are the values for the preference scores
calculated?]" placeholder — here's the answer to paste in:

> The preference scores are computed via a count-normalized, Bayesian-smoothed average: for each genre,
> the model computes the mean of the user's ratings within that genre, shrunk toward the neutral midpoint
> of the rating scale in proportion to how few ratings the user has in that genre (a Laplace/Bayesian
> smoothing prior). A user with zero ratings in a genre gets exactly the neutral score; a user with many
> ratings in a genre converges to their true observed average. This average is then passed through a
> learnable temperature-scaled sigmoid, and combined with a small, bounded residual signal from a
> secondary hidden pathway that captures genre-interaction nuance the simple average can't represent.

(Full derivation and the specific bug this replaced — an earlier version whose scores clustered almost
entirely at 0% or 100% because it used an *unnormalized sum* instead of a normalized average — is in
`docs/hcai_design_narrative.md`, Part II §1. That failure-and-fix story is strong Discussion material —
see below.)

The nudge-design paragraph ("to prevent the user from misjudging their preferences... only nudges the
preference scores") is **accurate and doesn't need changing** — it correctly describes both the original
intent and the current, actually-implemented mechanism (delta-based persisted overrides). Worth stating
in the paper that this design goal was preserved even though the underlying implementation mechanism
changed partway through — it's a nice continuity point.

### Section 4 — Experiment

1. **Resolve the within/between-subjects question first** (see TL;DR above) — this determines how the
   rest of the section, and the Results statistics, need to read.
2. The N=20 participant count, the group-A/group-B split, and the movie-rating task description (5–10
   movies, 0.5–5 stars in 0.5-star steps) are all consistent with the actual system as it now stands —
   no changes needed there.
3. **Terminology**: the paper now says "Treatment (T)" / "Control (C)"; the actual code and database
   still use `"O"` (Transparent) / `"N"` (Standard). Pick one naming scheme and use it consistently
   across the paper — if you keep T/C for readability, say once, explicitly, "referred to as Version O
   and Version N in the system implementation" so a reader who looks at the code or screenshots isn't
   confused by the mismatch.
4. **Figure 1** (participant background distribution, categorical: IT/Technik, Bildung/Forschung, etc.)
   — confirm with whoever built this figure whether these categories were manually coded from the
   free-text `degree_job` field the system actually collects, or whether a fixed-category dropdown was
   used somewhere not reflected in the current schema. If manually coded, say so in a footnote or the
   methods text ("job/field categories were derived via post-hoc coding of free-text responses") — this
   is a completely normal and legitimate step, but it should be stated, not implied.
5. The sentence "**The N system** would directly give the user recommendations" appears to be a leftover
   from the O/N naming — should read "the **C system**" to match the rest of the paragraph, which
   otherwise consistently uses T/C.
6. The section ends mid-sentence ("The questionnaire is loosely inspired by the SUS-Questionaire. It
   focuses" —) — needs finishing. Worth being more precise here than "loosely inspired by": the actual
   implementation is the **unmodified, standard 10-item Brooke (1996) SUS instrument** with the textbook
   scoring formula (odd items: response − 1; even items: 5 − response; sum × 2.5), identical for both
   conditions, plus three demographic questions asked once. That's a stronger, more citable claim than
   "loosely inspired" — use it if accurate to what was actually administered.

### Section 5 — Results

Currently empty. Once the within/between-subjects question (TL;DR #1) is settled, this section needs:
per-condition SUS score distributions (mean/SD at minimum), the statistical comparison appropriate to
the resolved design (paired t-test/Wilcoxon if truly within-subjects with linked sessions; independent
t-test/Mann-Whitney if between-subjects), and ideally a secondary behavioral measure beyond
self-report — the system already logs whether/how a user edited their profile (`has_edited`,
`profile_overrides` table), which could support a claim like "X% of Transparent-condition participants
used the editing feature at least once" as a usage-rate finding independent of the survey.

### Section 6 — Discussion and Conclusion

This is where the project's real strength should show up, and it's currently the weakest section
(unwritten). Two genuinely strong stories to use, both with a clear HCAI framing already worked out in
`docs/hcai_design_narrative.md`:

**Story 1 — the calibration failure.** An earlier version of the interpretable genre-bottleneck produced
scores that were technically interpretable (you could always point to what a dimension meant) but
practically meaningless (values clustered almost entirely at 0% or 100%, regardless of actual preference
strength). This is a good, citable example of a distinction worth making explicitly in Discussion:
**interpretability-as-labeling is not the same property as interpretability-as-informativeness** — a
representation can satisfy the first without satisfying the second, and this project has a concrete,
diagnosed, fixed example of exactly that gap.

**Story 2 — the popularity-bias correction that initially backfired.** An attempt to reduce
generic-popularity bias in recommendations (favoring personalized fit over broad appeal) was first
implemented as a pure "lift" ranking, which had an unanticipated failure mode: it could rank a genuinely
bad, broadly-disliked movie *above* a well-liked one, because the bad movie merely underperformed its own
(also low) baseline expectation by less than average. This is a strong methodological cautionary point:
**a well-motivated fairness/bias intervention needs its own empirical validation, not just a plausible
rationale** — exactly the kind of self-critical, iterative finding that distinguishes a rigorous paper
from a "we built X and it worked" report.

Both stories, plus four more (documented persistence of edits, and the explanation-faithfulness fix that
followed from the ranking change), are written up in full technical + HCAI-principle detail in
`docs/hcai_design_narrative.md`, Part II — that section is close to drop-in-ready Discussion prose.

### Related Work

No changes needed beyond the SHAP/LIME framing note under Introduction above.

---

## Terminology consistency checklist

Do one pass for these before final submission:

- **"Edibility" vs "Editability"** — the current title and body use "Editibility"/"Edibility"
  inconsistently across drafts seen so far; pick "**Editability**" (the standard English word) and use it
  everywhere, including the title.
- **O/N vs T/C** — pick one, state the mapping once if you use both anywhere.
- **HCAI vocabulary** — if you use the framing from `docs/hcai_design_narrative.md` (interpretability by
  construction, human-in-the-loop control, faithfulness, personalization vs. popularity bias,
  transparency about trade-offs, reversibility), use those exact terms consistently across Sections 3, 4,
  and 6 rather than rephrasing the same idea differently each time — a grader tracking your argument
  benefits from a stable vocabulary, and it reads as more deliberate.

## Sanity check before submitting

Read Section 3 and Section 4 side by side with `docs/hcai_design_narrative.md` Part IV one more time
after edits — that table is the fastest way to catch anything still-mismatched, and it'll stay useful
even if the paper keeps evolving after this pass.
