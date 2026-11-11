# Pre-registration: judge-rashomon full run on Arena human votes

**Status: DRAFT (2026-11-11).** This becomes frozen when Nishchay signs off and it is committed with the line "Frozen <date>" replacing this one. No Arena judge call is made before that. Changes after freezing go in Deviations, with a date and a reason.

Inputs:
- the pilot (`PREREG.md`, `RESULTS-pilot.md`)
- the Arena red-team and data-feasibility check (`judge-rashomon-research/06-arena-redteam.md`)
- the design dossier (`04-dossier.md`)

The pilot's verdicts informed only the feasibility choices listed under "What the pilot changed". No Arena verdict has been seen.

## Question
Take LLM-judge configs that agree with human preference about equally well. Do they reach different conclusions about which of two models is better? And does this happen more than chance and re-run noise would produce, especially when humans find the two models close?

## Data
- **Votes.** Pooled open LMArena human-preference releases: `lmarena-ai/arena-human-preference-55k`, `-100k` and `-140k` (140k is CC-BY-4.0). These are the versions already downloaded to `scratch_arena/`; the revision hashes are recorded at freeze time.
- **Filters**, applied in this order:
  1. English.
  2. Single-turn.
  3. Prompt of at least 20 tokens.
  4. Exact-duplicate prompts within a model pair dropped.
  5. Votes per user capped at 5 per model pair, if a user field exists.
- **Human label.**
  - Each vote is a winner, a tie or a both-bad; both-bad counts as a tie.
  - The human head-to-head win rate p_h for a model pair is computed on that pair's non-tie votes only. This is the estimand, not Bradley-Terry, because Arena samples battles non-uniformly (2504.20879).
- **Eligible model pairs.** At least 100 non-tie votes after filtering. The feasibility check measured about 135 pairs.
- **Items.**
  - Per eligible pair, draw 40 battles with non-tie human votes, uniformly at random with seed 0.
  - Split each pair's battles 20/20 by seeded hash into a **selection half** and a **measurement half**.
  - The human margin m = |p_h - 0.5| is computed from all of the pair's eligible votes, not just the 40 judged.

## Judge configs (8)
- **Judges.** 4 free-tier judges, each in 2 formats, with prompts, parsing and decoding identical to the pilot:
  - DeepSeek v4.1-flash (NIM)
  - Mistral-large-4 (NIM)
  - Gemma-4-31B (NIM)
  - Nemotron-3-ultra (OpenRouter `:free`, pinned to the Nvidia provider)
- **Formats.**
  - Pairwise: one call, A/B/tie.
  - Pointwise: two calls, 1-10 each, with a tie when the scores are equal.
- **Excluded.**
  - gpt-oss-20b: about 195 s per call on the only free provider, which is infeasible at this scale.
  - Claude CLI configs: kept at pilot size by Nishchay's budget rule.
- **Judge family**, used for the self-preference rule:

| Judge | Family |
|---|---|
| DeepSeek | deepseek |
| Mistral | mistral |
| Gemma | google, which also covers gemini, palm and bard contestants |
| Nemotron | nvidia, which also covers nemotron contestants |

- **Order bias.** The presentation order coin is balanced within each model pair. A 10% subset (seeded) is judged in both orders.
- **Logging.** Calls are interleaved across configs over time. Model ID, serving provider and date are logged on every call.

## Primary path (the one cell that carries the headline)
1. **Agreement.** Each config's agreement with humans is Cohen's kappa on the **selection half**, complete-case. A judge tie counts as a disagreement with a non-tie human vote.
2. **Eligibility.** A config needs coverage of at least 90% and failures of at most 5%, after exactly one transient retry. The same rule as pilot A5 and deviation 2026-11-10b.
3. **Rashomon set** R = configs whose paired kappa difference from the best is within **eps = 1 paired bootstrap SE** of that difference. The SE is per config and computed on the selection half. This replaces the pilot's fixed eps of 0.05, because the pilot showed a fixed eps is about 1 SE anyway and that membership is noisy.
4. **System conclusion.** On the **measurement half**, a config's conclusion for a model pair is the sign of (judge win rate - 0.5), with judge ties counted as 0.5. A win rate of exactly 0.5 means no conclusion, and the comparison is excluded.
5. **Self-preference.** A config pair is scored on a model pair only when neither config's judge family is one of the two contestants.
6. **Flip.** Two configs in R give opposite conclusions on a model pair. Unit = (config pair, model pair).
7. **Primary estimand.** The probability of a flip as a function of the human margin. It is fit as a logistic regression of the flip indicator on m, over all scored units inside R.
   - **Headline:** the fitted flip probability at m = 0.05 and at m = 0.20, plus the mean flip rate inside R.
8. **Uncertainty.** 2,000-draw bootstrap clustered by model pair: battles are resampled within each pair, and the pairs themselves are resampled. Selection (step 3) is re-run inside every draw, and m is recomputed from resampled votes, so the uncertainty in the human margin is carried through.

## Hypotheses
- **H1 (set exists):** R holds at least 2 configs.
- **H2 (beyond chance, primary):** the mean flip rate inside R beats both of these:
  1. the verdict-swap permutation null. 1,000 draws that re-run the whole procedure; one-sided p < 0.05.
  2. the re-run noise floor. The upper 95% bound of the flip rate between two re-runs of the same config, on the same units.
- **H3 (closeness):** the slope of flip probability on m is negative, with a one-sided 95% CI below 0.
- **H4 (validation buys stability, headline contrast):** the mean flip rate inside R is smaller than the mean across all eligible configs. Reported with a CI either way; no direction is predicted.
- **H5 (exploratory):** format (pairwise vs pointwise) explains more flips than judge model does.

**Pre-declared null.** If H2 fails, the headline reads: "configs validated against human votes disagree on system conclusions no more than chance or re-run noise, even for close pairs." That gets the same prominence as a positive result.

## Noise floor
The best pairwise and best pointwise configs in R are each re-run twice on the measurement half, with a cache-busting nonce. The re-run flip rate is computed on the same scored units as in step 6.

## Secondary and sensitivity (all labelled exploratory)
- **Descriptive classes:** clear (m > 0.1 and the human CI excludes 0.5), contested (the human CI lies within [0.4, 0.6]) and unresolved. No test is run on them, because the feasibility check found only about 11 contested pairs.
- **Confident flips:** both configs' bootstrap CIs on the win rate exclude 0.5, with opposite signs. Expected to be near 0 at 20 battles per pair, so it is not primary.
- **Fixed eps grid** from 0 to 0.20, as a curve.
- **No family exclusion:** all pairs.
- **Ties as wins of 0.5 versus ties dropped**, for judge ties.
- **Length:** a length-ratio covariate (answer A / answer B) added to the flip model, plus a markdown-density covariate.
- **Order:** both-orders agreement on the 10% subset.
- **Bradley-Terry ranking** instead of head-to-head win rate.
- **Item-level ambiguity** (Marx et al. 2020) inside R vs the re-run floor.

All sensitivity paths go into one labelled multiverse figure.

## Budget and stopping
- **Calls.** 135 pairs × 40 battles ≈ 5,400 items.
  - Each judge makes 3 calls per item (1 pairwise + 2 pointwise), so the 4 judges make about 65k.
  - The noise re-runs add about 4.3k, and the both-orders subset about 1.6k.
  - Total: **about 71k free-tier calls**.
- **Fallback.** If throughput projects more than 7 days, drop to 20 battles per pair (10/10 split) before any measurement-half verdict is looked at, and log it as a deviation.
- **Paid models.** None. If a provider's free tier disappears mid-run, its configs are judged on the A5 coverage rule as they stand, with no substitute model.
- **No interim looks** at flip rates. Progress checks report only coverage and failure counts.

## What the pilot changed (feasibility only)
- **Eps is defined in SE units**, because kappa SEs are about 0.05 and membership is noisy.
- **Raw flip rate is primary, and confident flips secondary,** because confident flips were 0 everywhere in the pilot and have little power.
- **The headline is set as a contrast with explicit nulls,** because a single point estimate moved from 0 to 0.033 when 7 items were dropped (gpt-oss addendum).
- **gpt-oss and the Claude CLI configs are dropped** (provider latency, budget).
- **Mistral is kept.** Its pilot failures came from the provider, and it is still subject to A5.

## Known limitations, stated in advance
- **Crowd votes reward style.** Kappa against them may select for alignment with style; the length covariate checks this.
- **Contamination is possible,** because Arena data is public. It would raise agreement, so it is conservative for H2.
- **Released Arena data skews toward older models,** so the conclusions cover the contestants in the releases, not the current leaderboard.
- **Free-tier providers can change models without notice.** IDs are pinned and logged, and any change is a deviation.

## Before freezing (checklist)
- [ ] Nishchay signs off on this draft.
- [ ] Re-check the 4 free model IDs and providers live.
- [ ] Re-run the novelty search (from 05-background-claims).
- [ ] Record the dataset revision hashes and the eligible pair count after filters.
- [ ] Code the primary path and run it on a synthetic dataset with a known flip structure. Confirm the permutation null and the logistic fit recover it, before any real call.

## Deviations
(none yet)
