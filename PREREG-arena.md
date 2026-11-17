# Pre-registration: judge-rashomon full run on Arena human votes

**Frozen 2026-11-11, before any Arena judge call was made.** Changes after this point go in Deviations, with a date and a reason.
- **Sign-off:** Nishchay signed off on the draft (commit c135972). He handed the pre-freeze decisions to Jet, who made the calls listed under "Pre-freeze changes".
- **Inputs:**
  - the pilot (`PREREG.md`, `RESULTS-pilot.md`)
  - the Arena red-team and feasibility check (`judge-rashomon-research/06-arena-redteam.md`)
  - the design dossier (`04-dossier.md`)
  - the novelty re-search (`05-background-claims.md`, 2026-11-11)
- **What the pilot informed:** its verdicts informed only feasibility choices, listed under "What the pilot changed". No Arena verdict has been seen.

## Question
Take LLM-judge configs that agree with human preference about equally well. Do they reach different conclusions about which of two models is better, more often than chance and re-run noise would produce? Is any such excess concentrated where humans find the two models close?

## Data
- **Votes.** Pooled open LMArena human-preference releases, at these Hugging Face revisions:

| Release | Revision |
|---|---|
| `lmarena-ai/arena-human-preference-55k` | `18c298340948c0e7f7727399fd459cca6ce0ca6f` |
| `lmarena-ai/arena-human-preference-100k` | `72e85b3ddc9c81bf7b659d6b03d4126dfd8fb34a` |
| `lmarena-ai/arena-human-preference-140k` | `6322995ab34d7c2693e3f47dd13fa5caa0789a74` |

  The 140k release is CC-BY-4.0.
- **Filters**, applied in this order in `src/arena_items.py`, with counts measured at freeze:
  1. English and single-turn: 158,518 battles. Language comes from the language field in 100k and 140k. 55k has no language field, so "English" there means at least 95% of letters are ASCII.
  2. Prompt of at least 5 whitespace words: 148,239.
  3. Both responses at most 4,000 words, which keeps free-tier context and latency bounded: 148,109.
  4. Exact-duplicate prompts dropped within a model pair, after whitespace normalisation: 147,744.
  5. No release has a per-user field, so the per-user vote cap is not applied.
- **Human label.**
  - Each vote is a winner or a tie; both-bad counts as a tie.
  - The pair's human win rate p_h is computed on its non-tie votes only. This head-to-head win rate is the estimand, not Bradley-Terry, because Arena samples battles non-uniformly (2504.20879).
  - The human margin is m = |p_h - 0.5|, computed from all of the pair's eligible votes.
- **Eligible pairs.** At least 100 non-tie votes: **80 pairs, from 50 contestant models**.
  - 26 pairs have m < 0.05, 42 have m < 0.1, and 16 have m ≥ 0.2.
  - The full list is in `data/arena_pairs.json` (sha256 prefix `4c9f88e1d7a47aeb`).
- **Items.**
  - Per pair, 40 non-tie battles are drawn uniformly at random (seed 0).
  - They are split 20/20 by seeded hash into a **selection half** and a **measurement half**: 3,200 items, in `data/arena_items.json` (sha256 prefix `99a34d11b806541e`).
  - The presentation order coin is balanced within each pair: exactly 20 of the 40 are swapped.
  - A seeded 10% subset (320 items) is judged pairwise in both orders.

## Judge configs (6)
- **Judges and providers.** 3 free-tier judges, each in 2 formats. Prompts, parsing and decoding are identical to the pilot (`src/run_judges.py`).

| Judge | Model | Provider |
|---|---|---|
| DeepSeek | `deepseek-ai/deepseek-v4.1-flash` | NIM |
| Mistral | `mistralai/mistral-large-4-1t-a52b` | NIM |
| Nemotron | `nvidia/nemotron-3-ultra-550b-a55b` | NIM, directly (in the pilot, OpenRouter `:free` pinned to the Nvidia provider) |

- **Formats.**
  - Pairwise: one call, A/B/tie.
  - Pointwise: two calls, 1-10 each, with a tie when the scores are equal.
- **Excluded.**
  - Gemma-4-31B: unreachable on NIM at freeze time (see Pre-freeze changes).
  - gpt-oss-20b: about 195 s per call.
  - Claude CLI configs: kept at pilot size.
- **Judge family**, used for the self-preference rule. A contestant belongs to a judge's family if its name starts with:

| Judge | Contestant prefixes |
|---|---|
| deepseek | `deepseek` |
| mistral | `mistral`, `mixtral` |
| nemotron | `nemotron` |

  The 80 pairs include 1 deepseek contestant (deepseek-v3-0324), 4 mistral or mixtral contestants and no nemotron contestants.
- **Logging and failures.**
  - Calls are interleaved across configs over time.
  - Model ID, serving provider and date are logged on every call.
  - After rep 0, exactly one retry of transient failures (timeout, connection, 429, 5xx), as in pilot deviation 2026-11-10b.

## Primary path (`src/arena_analyze.py`)
1. **Eligibility.** A config needs coverage of at least 90% and failures of at most 5% after the one retry.
2. **Agreement.** Cohen's kappa against human votes on the **selection half**, complete-case across eligible configs. A judge tie counts as a disagreement with a non-tie human vote.
3. **Rashomon set.** R = the best config, plus every config whose kappa gap to the best is at most 1 SE of that paired gap. The SE is a cluster jackknife over model pairs, on the selection half. The pilot showed a fixed eps is about 1 SE anyway, and that membership is noisy.
4. **System conclusion.** On the **measurement half**, complete-case, a config's conclusion for a pair is the sign of (judge win rate for the first model - 0.5), with judge ties counted as 0.5. Exactly 0.5 means no conclusion, and the comparison is excluded.
5. **Self-preference.** A config pair is scored on a model pair only when neither config's judge family is one of the two contestants.
6. **Flip.** Two configs in R give opposite conclusions. Unit = (config pair in R, model pair).
7. **Estimands.**
   - The **mean flip rate inside R**.
   - A logistic regression of the flip indicator on m, giving fitted flip probabilities at m = 0.05 and m = 0.20 and the slope.
8. **Null.** A **joint verdict permutation** on the measurement half, with 1,000 draws. On every item, the verdicts of the configs in R are shuffled among themselves, and the whole measurement step is recomputed. R is fixed from the untouched selection half.
9. **Uncertainty.** A 2,000-draw cluster bootstrap. Each draw resamples the pairs, then the battles within each pair's two halves, and draws m from Binomial(n_votes, p_h). Selection (step 3) is re-run inside every draw.

## Hypotheses
- **H1 (set exists):** R holds at least 2 configs.
- **H2 (beyond chance, primary):** both of the following hold:
  1. The mean flip rate inside R has a permutation p below 0.05, one-sided.
  2. It exceeds the upper 95% bound of the re-run noise floor (see Noise floor).
- **H3 (closeness beyond chance):** the observed slope of flip probability on m is more negative than the permutation null's slopes, with one-sided p below 0.05.
  - Under the null, flips fall with m from sampling alone, so a negative slope by itself is not evidence. The dry run confirmed this.
- **H4 (validation buys stability, headline contrast):** the mean flip rate across all eligible configs minus the mean inside R, with a bootstrap CI. No direction is predicted.
- **H5 (exploratory):** whether format (pairwise vs pointwise) or judge model explains more flips.

**Pre-declared null.** If H2 fails, the headline reads: "configs validated against human votes disagree on system conclusions no more than chance or re-run noise, even for close pairs." That gets the same prominence as a positive result.

## Noise floor
- **What is re-run.** The best pairwise and best pointwise configs among the eligible ones are each re-run twice on the measurement half, with a cache-busting nonce.
- **What is measured.** The flip rate between rep 0 and each re-run, on the same scored units.
- **The bound.** The upper 95% bound comes from a bootstrap over those units. H2 uses the larger of the two configs' bounds.

## Dry run on synthetic data (done before freezing)
The dry run (`src/arena_synth.py`; results in `results/arena_synth_dryrun.json`) used the real items and human labels with synthetic verdicts for the 6 configs: 40 runs per scenario, 200 bootstrap draws and 200 permutations.

| Scenario | Permutation p < 0.05 | H3 rejections | H2 passes |
|---|---|---|---|
| **Null:** configs exchangeable, each follows the human vote with p = 0.75 | 2.7% | 2.5% | 0% |
| **Planted:** two camps prefer opposite models on pairs with m < 0.1, equally accurate against humans | 70% of runs with R ≥ 2 | 37.5% | 57.5% |

- **Calibration.** The tests are calibrated under the null.
- **Planted runs with R < 2.** 7 of 40 runs had a set with fewer than 2 configs.
- **Power.** For an effect this strong, H2 power is about 0.6 and H3 power about 0.4. We record this now so a null result cannot later be over-read as "no multiplicity".
- **Bug fixed by the dry run.** The first version of the null used independent pairwise swaps, as the pilot did. It rejected in about 20% of null runs, because it breaks the correlation between config pairs that share a config. The joint permutation above replaced it.

## Secondary and sensitivity (all labelled exploratory)
- **Prompt-length and vote-threshold rule (pre-registered).** Restrict to judged items whose prompt has at least 10 words. Recompute each pair's human margin from its at-least-10-word votes. Keep pairs with at least 60 such votes. Re-run the primary path on that subset. No extra judge calls.
- **Descriptive classes:** clear (m > 0.1 and the human CI excludes 0.5), contested (the CI lies within [0.4, 0.6]) and unresolved. Descriptive only.
- **Confident flips:** both configs' bootstrap CIs on the win rate exclude 0.5, with opposite signs.
- **Fixed eps grid** from 0 to 0.20, as a curve.
- **No family exclusion.**
- **Judge ties dropped** instead of counted as 0.5.
- **Length:** a length-ratio covariate (answer A / answer B words) and a markdown-density covariate added to the flip model.
- **Order:** both-orders agreement on the 320-item subset.
- **Bradley-Terry ranking** instead of head-to-head win rate.
- **Item-level ambiguity** (Marx et al. 2020) inside R vs the re-run floor.

All sensitivity paths go into one labelled multiverse figure.

## Budget and stopping
- **Calls.**
  - 3,200 items × 3 calls per judge × 3 judges = 28.8k.
  - Noise re-runs: 2 configs × 2 reps × 1,600 measurement items, about 9.6k.
  - Both-orders subset: 320 × 3, about 1k.
  - Total: **about 39k free-tier calls**.
- **Slowest provider.** DeepSeek on NIM took 228 s for one test call at freeze time. If its projected finish time exceeds 7 days, the fallback is cut in this order:
  1. Its noise re-runs are dropped. The noise floor then uses the other configs.
  2. If that is still not enough, all judges drop to 20 battles per pair (10/10).
- **Fallback rule.** Either fallback is decided from throughput alone, before any measurement-half verdict is looked at, and is logged as a deviation.
- **No paid models and no substitute models.** A config that loses its provider mid-run is judged on the eligibility rule as it stands.
- **No interim looks** at flip rates. Progress checks report only coverage and failure counts.

## Pre-freeze changes from the signed-off draft (2026-11-11; decided before any Arena verdict existed)
1. **Prompt minimum lowered from 20 to 5 words.** With 20 words, only 11 pairs had at least 100 non-tie votes, below the 40-50 the red-team asked for. Measured counts:

| Minimum prompt length | Pairs with ≥100 votes | Pairs with ≥60 votes |
|---|---|---|
| 20 words | 11 | 40 |
| 10 words | 39 | 184 |
| 5 words | 82 | 401 |

   After de-duplication, 80 pairs remain at 5 words. Precise human margins (at least 100 votes, SE about 0.05) were judged more important than prompt substance. 5 words still removes greetings. The 10-word / 60-vote rule became the sensitivity analysis above. Decision by Jet, delegated by Nishchay.
2. **Response cap added.** Responses longer than 4,000 words are excluded, which removes 130 battles.
3. **Nemotron moved from OpenRouter to NIM.** It is the same model, served by Nvidia in both cases. OpenRouter's free tier caps it at about 1,000 calls per day (it returned 429 `free-models-per-day-high-balance` at freeze time), which would take more than two weeks at this scale.
4. **Gemma-4-31B dropped.** It timed out on NIM on 3 tries on 2026-11-11 (two at 120-180 s, then 504 at 302 s, then timeout at 240 s). Under the rule "freeze without it if it is still down after one more try", it is excluded, and config count falls from 8 to 6.
5. **eps defined via a cluster-jackknife SE** over model pairs, not the paired bootstrap SE. The SE has to be recomputed inside each of the 2,000 bootstrap draws, and the jackknife makes that cheap.
6. **Joint permutation null** replaces the pairwise swap (see Dry run).
7. **H3 is tested against the permutation null**, not against slope 0 (see Dry run).

## What the pilot changed (feasibility only)
- **eps is defined in SE units.**
- **The raw flip rate is primary, and confident flips secondary**, because confident flips were 0 everywhere in the pilot.
- **The headline is a contrast with explicit nulls**, because a pilot point estimate moved from 0 to 0.033 when 7 items were dropped.
- **gpt-oss and the Claude CLI configs are excluded.**

## Known limitations, stated in advance
- **Three judges, of which two (DeepSeek, Mistral) have family contestants.** Format and judge are therefore only partly separable.
- **Power is moderate** (see Dry run).
- **Crowd votes reward style**, so kappa against them may select for alignment with style. The length covariate checks this.
- **Contamination is possible**, because Arena data is public. It would raise agreement, so it is conservative for H2.
- **The released Arena data skews toward older contestants.**
- **Free-tier providers can change models without notice.** IDs are pinned and logged, and any change is a deviation.

## Deviations
- **2026-11-11a (before any Arena verdict was analysed): Nemotron on NIM retries empty replies.**
  - **What was observed.** In the 3-item smoke test, `nvidia/nemotron-3-ultra-550b-a55b` on NIM often stopped after 6-11 reasoning tokens with `finish=stop` and no content. On one item, 2 of 3 identical requests came back empty.
  - **Why OpenRouter is not an option.** Its free tier is still capped at about 1,000 calls per day.
  - **Decision** (Jet, under Nishchay's delegation):
    - Nemotron stays on NIM, with 3 requests in flight and its own process paced to at most 8 requests per minute.
    - Before the standard parse retries, an empty reply is re-sent up to 3 more times with backoff (10, 20 and 40 s). Every retry is logged to `logs/nemotron_empty_retries.log`.
  - **Why this doesn't bias results.** An empty reply contains no verdict, so it is a transport failure, not a judgment. Re-sending the identical request cannot select for any particular verdict. Every item is retried by the same rule, regardless of content.
  - **What happens if replies stay empty.** Replies still empty after retries count as failures under the existing eligibility rule (at most 5%). If Nemotron exceeds that cap, it is dropped as Gemma was, and recorded here. It is never restricted to the items that happened to succeed.
  - **What was discarded.** The 3 smoke-test Nemotron entries were moved to `cache_arena_failed/smoke_nim/` and not used.
  - **No change for DeepSeek and Mistral.** They started at 06:44 PT on 2026-11-11 under the frozen settings.
  - **Same day: endpoint diagnosed as broken server-side, so Nemotron is paused.** NIM's Nemotron-3-Ultra stops generating after 0-50 tokens under every setting tried:
    - the frozen settings
    - `enable_thinking` set to false
    - `force_nonempty_content`
    - temperature 0.01 and 1.0
    - streaming
    - NVIDIA's own documented sample request, which came back empty 6 out of 6 times

    With retries, 14 of the first 15 items did get a verdict. But each one took about 5 requests, and at that rate the 40-requests-per-minute account limit cannot finish about 9.9k calls. The process was stopped. `nemotron_wait.sh` re-runs the documented sample every 2 h (`src/nim_health.py`). Once at least 5 of 6 come back non-empty, it resumes Nemotron rep 0 under the rules above. Responses already cached are kept, since the same rule applies to every item.
  - **Mistral concurrency raised.** Mistral's worker count went from 6 to 16 (throughput only; the cache meant nothing was re-judged).
- **2026-11-11b (before any measurement-half verdict was looked at): gpt-4o-mini pairwise added as a judge.**
  - **Why.** With Nemotron paused and Gemma excluded, only 4 configs remained.
  - **Approval.** Nishchay approved a paid OpenAI judge with a $2 cap.
  - **Config.** `gpt-pair` = `gpt-4o-mini-2024-07-18` through the OpenAI Batch API.
    - Same pairwise prompts, parsing, balanced order coin and both-orders subset as every other config.
    - Temperature 0, max_tokens 1024. The 1024 matches the pilot PREREG text and bounds the worst-case cost.
  - **Pairwise only:** 3,520 requests, worst case $1.36 at batch prices ($0.075 per million input tokens, $0.30 per million output, checked 2026-11-11).
  - **Family rule:** contestants named `gpt*` or `chatgpt*` are excluded for this judge.
  - **Same rules as everyone else.**
    - It enters the eligibility rule, the selection and the set like any other config.
    - If it is chosen as the best pairwise config for the noise floor and the re-runs would push spending past $2, the noise floor uses the next-best pairwise config instead. This rule is fixed now, before any kappa is computed.
    - Transient batch failures get the same one retry.
  - **If the account runs out of credit,** the affected requests are recorded as failures and the eligibility rule decides. A Codex-subscription fallback would be a further, separate deviation.
- **2026-11-11c (during rep 0; no verdict analysed): rate-limit storm, concurrency lowered.**
  - **What happened.** With DeepSeek at 24 workers and Mistral at 16, NIM returned HTTP 429 even after 6 backoff attempts. In the first ~2 hours, 185 of the 444 DeepSeek and Mistral requests ended as cached transient errors: 151 were 429s, and the rest were 502/503/504 or connection errors.
  - **Cause.** Our own concurrency, not the items.
  - **Fix.**
    - The process was stopped and restarted with DeepSeek at 8 workers, Mistral at 6, and paced to at most 20 requests per minute.
    - The 185 transient-error entries were moved to `cache_arena_failed/rep0/` and re-queued.
  - **This uses up their one retry.** The post-rep-0 retry pass skips any entry already in `cache_arena_failed/`, so each of these 185 entries is retried exactly once, as for every other item.
  - **Why this doesn't bias results.** The errors carry no verdict, so re-sending them does not depend on content.
- **2026-11-11d (during rep 0; no verdict analysed): a 429 is not an attempt.**
  - **What happened.** Even at 20 requests per minute, NIM kept returning HTTP 429.
  - **What changed.**
    - A 429 means the request was never served, so it is no longer counted as an attempt or cached as a failure.
    - The pacer now re-sends the request and backs the whole process off: 30 s, doubling, up to 10 min.
  - **Effect on cached entries.**
    - 4 cached 429 entries from rep 0 were deleted, so those requests run again.
    - 151 entries that 2026-11-11c had moved to `cache_arena_failed/` as 429s were removed from there, so they no longer count as used retries.
    - Only real served failures (5xx, timeouts, connection errors) keep counting toward the one-retry rule and the 5% failure cap.
- **2026-11-11e (no verdict analysed): gpt-4o-mini pointwise added (`gpt-point`).**
  - **Why.** NIM throughput fell to about 0.7 calls per minute, which would put DeepSeek and Mistral at about 19 days. Nishchay approved adding the pointwise format for gpt-4o-mini.
  - **Config.**
    - Same pointwise prompts and parsing as every other config.
    - Temperature 0 and max_tokens 600, through the Batch API: 6,400 requests.
  - **Why max_tokens 600.** It keeps the worst case at $1.44, under the $2 total cap with $0.50 already spent. On gpt-pair, reply lengths had a p99 of 394 tokens and a maximum of 517.
  - **Rules.** Same family rule, eligibility rule and noise-floor budget rule as gpt-pair.
- **2026-11-12f / 2026-11-13f (during rep 0; no verdict analysed): timeouts and workers.**
  - **What happened.**
    - 2026-11-12: all NIM runs stalled for about 3 hours, with DeepSeek calls hanging under a 600 s timeout. The runs were restarted with a 120 s timeout.
    - The 120 s timeout was shorter than DeepSeek's typical reply time (about 230 s), so about 35% of DeepSeek calls timed out. There were 0 HTTP 429s over the same period, so the request rate was far from NIM's limit.
  - **What changed (2026-11-13).**
    - Timeout set to 400 s.
    - Workers raised: DeepSeek 8 → 20, Mistral 6 → 14. Nemotron unchanged at 3.
    - The process-wide 429 backoff from 2026-11-11d stays in place.
  - **Why this doesn't bias results.** Timeouts carry no verdict and fall under the one-retry rule as before. Only throughput changes.
- **2026-11-13g (no verdict analysed): HTTP 404 counted as transient.**
  - **What happened.** NIM briefly answered "model not found" (404) for Mistral during otherwise normal operation, affecting about 50 entries so far.
  - **What changed.** A 404 is treated like a 5xx: it carries no verdict, gets the one retry, and doesn't count toward the 5% failure cap unless the retry also fails.
- **2026-11-13h (no verdict analysed): staged release.**
  - **Why.** Free-tier NIM throughput puts complete DeepSeek and Mistral rep 0 several days out. Nishchay asked for a faster first release.
  - **Release 1 (interim, labelled exploratory).**
    - The pre-registered pipeline runs unchanged on the configs whose rep 0 is complete at the cutoff: gpt-pair, gpt-point, nemotron-pair and nemotron-point.
    - Completion is driven by server latency, not by verdict content.
    - Release 1 makes no confirmatory H2 or H3 claim.
  - **Release 2 (confirmatory).** The full pre-registered analysis with all configs, once DeepSeek and Mistral finish. Release 1 numbers are not used to change anything in Release 2.
- **2026-11-16i (no measurement-half verdict analysed): final release without completing NIM runs.**
  - **Why.** NIM throughput stayed far below plan: DeepSeek pointwise reached 4,635 of 6,400 items after 5 days, and the one-retry pass was queuing about 2,000 transient failures. Nishchay asked to park the remaining NIM work and finalize with what is complete. The decision rests on throughput alone.
  - **What is parked.**
    - DeepSeek pointwise rep 0 (72% coverage).
    - The one-retry pass for Mistral and DeepSeek. 133 of about 2,000 queued calls had been retried when it stopped; the rest keep their transient failures, which count as failures.
    - The noise-floor re-runs (none started).
  - **How the configs are judged.** Unchanged eligibility rule (coverage ≥ 90%, failures ≤ 5%), as the budget section states for a config that loses its provider mid-run. Incomplete configs are left out.
  - **Effect on hypotheses.**
    - H2 is tested against the permutation null only. Its noise-floor condition is not tested, so H2 cannot be confirmed in full.
    - H1, H3 and H4 run as pre-registered on the eligible configs.
  - **What was seen before this decision.** Coverage and failure counts, Release 1's gpt-only numbers, and the dry-run noise pick (config names, chosen by selection-half kappa). No measurement-half verdict of any NIM config.
  - **This replaces Release 2 from 2026-11-13h** as the final Arena analysis.
