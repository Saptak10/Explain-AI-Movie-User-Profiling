# Architecture and mechanism changes

This note summarizes what changed in the system since the original design, and why, for use in the
Methodology section of the paper. Four independent changes, in the order they were made:

1. **Genre-bottleneck calibration** (model architecture) — fixes genre-affinity percentages clustering
   near 0%/100%.
2. **Recommendation ranking: personalization lift + quality floor** (inference-time ranking) — fixes
   certain movies (documentaries were the clearest case) being recommended regardless of a user's actual
   profile.
3. **Persisted profile edits** (interactivity / data model) — makes manual genre-preference edits a
   durable part of a user's profile instead of a one-off, single-response adjustment.
4. **Lift-aware XAI rationale** (explainability) — makes the natural-language "why was this recommended"
   explanation describe the mechanism that *actually* ranks recommendations (change #2), instead of a
   generic fallback message whenever genre-dominance alone doesn't explain a movie.

None of these required retraining except #1, which needed a fresh checkpoint because it changes the
model's forward pass; #2–#4 are inference-time/application-layer changes over a fixed, already-trained
model.

## 1. Genre-bottleneck calibration

### The problem the old design had

The original (and first-revision) architecture built each user's genre-affinity score via a **hard,
weight-clipped prior injection**:

- `encoder_l1`'s first 19 rows were seeded with the raw 0/1 genre-membership mask (`target_genre_matrix`)
  and `encoder_l2`'s first 19 columns were seeded with the identity matrix, so that — before any training
  — the genre bottleneck exactly reproduced the sum of a user's ratings within each genre.
- After every optimizer step, `apply_weight_clipping()` clamped those same weights back to within
  `epsilon = 0.15` of that hard prior ("elastic-leash" regularization), so gradient descent was never
  allowed to move the anchored weights far from the raw genre mask.
- A first revision added a single learnable temperature scalar dividing the logit before the sigmoid,
  intended to desaturate the resulting activations.

Two problems fell out of this design, confirmed by (a) loading the trained checkpoint's actual weights
and (b) running synthetic-user simulations against them:

1. **Count-dependence.** Because the anchored pathway is a *sum*, not an *average*, the pre-sigmoid logit
   scaled roughly linearly with how many movies a user had rated in a genre. A synthetic-user simulation
   against the real trained weights showed logits of ≈ −0.6 to 3.6 for a 1-rating user, ≈ 35–44 for a
   10-rating user, and ≈ 600–700 for a 200-rating user in the same genre. No single global temperature
   can reconcile that range: a value large enough to de-saturate the 200-rating case flattens the
   1-rating case to ~50% (no signal at all), and vice versa. Consistent with this, the learned temperature
   only drifted from its 5.0 initialization to 6.6 over 10 epochs of training — the gradient signal for it
   vanishes once activations are already saturated, which is most of them.
2. **An unregularized "free" pathway.** Only the first 19 of the hidden layer's 128 units were
   prior-anchored and weight-clipped. The remaining 109 units fed the same genre-bottleneck logit with no
   constraint of any kind. Inspecting the trained checkpoint directly showed this free pathway's weights
   already exceeded the entire anchored deviation budget in magnitude, contributing uncontrolled logit
   swings on the order of ±4 to ±106 in the simulation — a second, independent source of saturation that
   no amount of temperature tuning on the anchored path could address.

Net effect: genre-affinity percentages clustered heavily near 0% and 100% for most users, regardless of
the temperature fix, because the underlying signal was fundamentally an unbounded, unnormalized sum
rather than a bounded, meaningful average.

### The new design

The genre-bottleneck logit is now built from two additive parts, computed in `forward_standard`:

**1. An anchored signal — closed-form, not learned.**

```
rated_mask   = (x != 0)                                  # which movies the user rated
genre_sum    = x            @ target_genre_matrix.T        # Σ rating, per genre
genre_count  = rated_mask   @ target_genre_matrix.T        # # ratings, per genre
genre_mean   = (genre_sum + count_smoothing * rating_midpoint) / (genre_count + count_smoothing)
anchored_logit = (genre_mean − rating_midpoint) / genre_temperature
```

`target_genre_matrix` is the same fixed genre-membership buffer used before, but it is now read directly
at every forward pass rather than used only to *initialize* a weight matrix that then drifts under
training. `genre_mean` is a **Bayesian/Laplace-smoothed average** rating per genre — `count_smoothing`
(a small constant, 1.5) acts as a pseudo-count anchored at the neutral midpoint of the rating scale
(`rating_midpoint = 2.5`), so:

- 0 ratings in a genre → `genre_mean = 2.5` exactly → `anchored_logit = 0` → 50% (fully neutral), by
  construction, for any user regardless of how many ratings they have elsewhere.
- A handful of ratings → `genre_mean` sits between the prior (2.5) and the observed average, shrunk
  proportionally to how little evidence there is (the fewer the ratings, the closer to neutral).
- Many ratings → `genre_mean` converges to the true observed average, with negligible shrinkage.

Because this is a closed-form computation over a *rate* (an average), not a *sum*, it is **count-invariant
by construction** — the same `genre_temperature` scalar now meaningfully calibrates the signal regardless
of whether a user has 1 or 200 ratings in a genre, which was structurally impossible under the old
sum-based design. `genre_temperature` remains learnable, but its job changed from "try to rescale an
unbounded quantity" to "calibrate the sharpness of a signal that is already on a bounded, meaningful
scale" — a much easier and better-posed optimization problem.

**2. A bounded residual signal — still learned, but capped.**

```
residual_logit = tanh( encoder_l2( ReLU( encoder_l1(x) ) ) ) * residual_scale
```

The same two-layer hidden pathway remains, for learned nuance (genre interactions, non-linear effects
the closed-form average can't capture) — but its output is now passed through `tanh` and multiplied by a
learnable `residual_scale` (initialized to 0.5, clamped to `[0, 2.0]` at use-time). This bounds the
residual's contribution to a fixed range regardless of the magnitude of the underlying weights, so it can
add nuance but can no longer dominate or destabilize the anchored signal the way the old fully-unconstrained
free pathway could. A complementary L2 penalty on these weights (`get_semantic_loss`, now repurposed —
see below) discourages the residual from growing large in the first place.

**3. Final activation, unchanged in form:**

```
latent_profile = sigmoid(anchored_logit + residual_logit)
```

### What this made obsolete

- `_inject_prior_knowledge()` — removed. There is no longer a weight matrix to seed toward the genre
  prior; the prior *is* the direct computation in part 1.
- `apply_weight_clipping()` — removed, and the corresponding call in the training step (`losses.py`,
  `train_step`) removed along with the now-meaningless `epsilon_clip` hyperparameter (previously threaded
  through `config.py`, `train.py`'s CLI, and `ai_service.py`). There is nothing left to clip toward a prior
  — the anchored signal cannot drift, by construction, so the elastic-leash mechanism has no remaining job.
- `get_semantic_loss()` — repurposed rather than removed. It previously measured squared drift of the
  anchored weights from their prior (a quantity that no longer exists); it now measures an L2 penalty on
  the residual pathway's weights, serving the complementary role of discouraging the residual from growing
  large, alongside the `tanh`/`residual_scale` bound in the forward pass.

`forward_interactive` (the manual genre-slider override path) and `extract_taste_profile` (percentage
extraction for the UI) are unchanged in structure — `forward_interactive` bypasses the encoder entirely
and was never affected by this mechanism, and `extract_taste_profile` still just reads `latent_profile`
and scales by 100.

### Framing for the paper

If useful phrasing for the methodology section: the original design used **hard-constrained weight
injection with post-hoc elastic regularization** to keep a *learned* bottleneck faithful to genre
semantics — an approximation that turned out to leave the actual signal an unnormalized, count-dependent
sum, with a real regularization gap in the untouched two-thirds of the hidden layer. The revised design
instead computes the semantically-anchored component **exactly**, as a closed-form, Bayesian-smoothed,
count-invariant average — removing the need for weight injection or clipping entirely for that
component — and retains a small, explicitly bounded residual pathway for learned nuance. The practical
motivation was empirical (genre percentages were bimodal near 0%/100%, not the smooth, information-bearing
spread the "Human-Centered" taste-profile explanation is meant to show a user), and the fix follows
directly from diagnosing *why*: the old mechanism conflated "faithful to genre semantics" with "faithful
to a fixed weight matrix," when the actual failure mode was a scale/normalization problem the weight
matrix framing couldn't expose or fix.

## 2. Recommendation ranking: personalization lift and a quality floor

### The problem: a genre-independent decoder bias

The decoder (`decoder_l1` → ReLU → `decoder_l2`) takes only the 19-dimensional genre latent as input, but
`decoder_l2` is an ordinary `nn.Linear` with a **per-movie bias term** — one learned scalar per movie
(~87,000 of them), entirely independent of genre. This bias captures a movie's *generic* appeal: how
highly it tends to be predicted regardless of a specific user's genre profile.

Diagnosed empirically by running the decoder on a fully genre-neutral input (0.5 for every genre — the
"no distinguishing signal at all" point under the new calibration in §1, since `sigmoid(0) = 0.5`) and
inspecting which movies score highest with *zero* personalization applied. Several documentaries ranked in
the global top-20 by this genre-neutral score, alongside universally-beloved films — both groups share a
high generic bias, for different underlying reasons (documentaries: a small, self-selected, enthusiastic
rater population with few negative ratings, since casual viewers rarely watch or rate documentaries at
all; universally-beloved films: genuinely broad appeal). The model cannot distinguish these two cases from
its own output alone, since both present as "high score regardless of profile" — but ranking purely by raw
predicted score lets *both* dominate every user's recommendations regardless of whether that user's own
ratings give any reason to expect they'd specifically enjoy that title.

### First attempt (rejected): unconditional lift ranking

The natural fix is to rank by **personalization lift** — `predicted_score − baseline_score`, where
`baseline_score` is the same genre-neutral decoder output described above, cached once per loaded model
(`AIService._compute_baseline`). This isolates the genre-driven, personalized component of a prediction
from the generic bias term.

Ranking by *unconditional* lift has a real failure mode, however, confirmed by testing: a movie that is
**bad for essentially everyone** (very low baseline) can still show *positive* lift if a user's specific
prediction is merely "less bad than average" for it — e.g. baseline 0.7, this user's prediction 1.7, lift
+1.0. Unconditional lift-ranking surfaced near-unwatchable movies (e.g. *Birdemic: Shock and Terror*) ahead
of genuinely well-predicted, broadly-loved films whose lift is near zero precisely *because* they're
already correctly predicted well for almost everyone (predicted ≈ baseline, both high). Lift alone
conflates "unusually good fit for you" with "merely less-bad-than-expected," which are not the same thing.

### Final design: quality floor, then lift

```
MIN_SCORE_FOR_LIFT_RANKING = 3.0   # on the 0-5 rating scale

if predicted_score >= MIN_SCORE_FOR_LIFT_RANKING:
    rank_score = predicted_score − baseline_score        # ranked by lift among quality-passing candidates
else:
    rank_score = predicted_score − 100                   # always sorts below every quality-passing candidate
```

Candidates are ranked by lift **only** among those whose raw predicted score already clears a quality floor
(3.0/5); everything below the floor is ranked at the bottom by raw score, so lift can never promote a
poorly-predicted movie above a well-predicted one. This resolves both original problems: within the
quality-passing set, personalized-fit movies now outrank generically-high-bias movies with near-zero lift
(the documentary case), while the floor prevents genuinely bad movies from exploiting a low baseline (the
*Birdemic* case). The number **displayed** to the user is always the raw predicted score (a 0–5 star
prediction, the number actually meaningful to them), never the lift — lift is a ranking criterion only.
Implemented in `AIService._top_n_from_scores`, used by both `get_recommendations` and
`create_personalized_profile` (the latter computes its own baseline against the fine-tuned model clone,
since a personalization fine-tune shifts the decoder's bias terms too).

## 3. Persisted profile edits

Genre-preference edits (the boost/suppress buttons on the Profile and Recommend pages) previously applied
only to the single response they were submitted with — revisiting the Profile page or requesting new
recommendations reverted to the pure AI-inferred profile, discarding any edit a user had made. This meant
the "Human-Centered" interactive-editing feature had no lasting effect: a user's stated preference
correction never actually changed what they were recommended beyond one screen.

Edits are now persisted server-side as **deltas** (not absolute values) in a new `profile_overrides` table,
keyed by `(user_id, genre)`. Storing a *delta* rather than an absolute override value keeps an edit
meaningful as the AI-inferred profile itself continues to shift with new ratings — "boost this genre by
+0.25" stays interpretable regardless of what the underlying AI estimate currently is, whereas a stored
absolute value would silently go stale. Every read path that produces "the user's profile" —
`GET /api/profile`, `POST /api/recommend`, `POST /api/explain` — now merges these persisted deltas onto the
AI-inferred latent profile before use (`AIService._merge_overrides_into_latent`), so an edit affects every
subsequent screen and recommendation until the user changes it again or explicitly resets
(`DELETE /api/profile/overrides`, "Reset to AI Profile" in the UI). `POST /api/recommend/edited-profile`
(the Apply action itself) both saves the submitted deltas and returns recommendations computed with them
in the same request.

This also simplified the interactive-override mechanism itself: `forward_interactive` was previously called
standalone with a hand-built override vector representing the *entire* genre bottleneck, which meant any
genre the UI didn't explicitly include in a request was implicitly zeroed rather than left at the
AI-inferred value (a real bug, fixed alongside the persistence work). The corrected design always starts
from the encoder's own `forward_standard` output and merges only the specifically-overridden genres on top,
before decoding — never bypassing the encoder's inference for genres the user hasn't touched.

## 4. Lift-aware XAI rationale

`generate_soft_rationale` (`backend/app/ai/xai.py`) produces the natural-language "why was this
recommended" explanation. Its genre-dominance logic (a genre counts as the explanation if it's one of the
target movie's own genres *and* this user's activation for it is above their own mean activation across
all genres — itself a change from an earlier fixed `0.6` absolute threshold, made obsolete by §1's
calibration change) covers the case where a specific genre visibly drives a recommendation.

Once §2's lift-based ranking landed, that dominance check no longer covers *why a movie was rankable at
all* — the two questions are genuinely different mechanisms, only one of which the old code could explain.
A movie can rank highly with no single dominant genre for two distinct reasons that deserve two distinct
explanations: (a) high lift — it fits this user's specific rating pattern beyond generic appeal, even
though no one genre stands out, or (b) low lift but a high absolute score — it's broadly well-liked, and
this user's ratings simply don't contradict that. The old fallback ("recommended based on general
collaborative filtering patterns across similar users") collapsed both into one uninformative message that
described neither mechanism.

The fallback now takes the same `predicted_score` and `baseline_score` quantities §2 ranks with (computed
in `AIService.explain_movie` from the same, override-adjusted forward pass used for the recommendation
itself) and reports whichever of (a) or (b) actually applies, citing the numeric lift or absolute score
directly — e.g. *"predicted 4.1/5 for you specifically — 0.4 stars above what a typical viewer would get"*
for case (a), or *"widely well-liked regardless of genre profile (predicted 4.1/5)"* for case (b). The
explanation now always names the actual mechanism that produced the recommendation, rather than falling
back to a message that was accurate for neither.
