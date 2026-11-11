# judge-rashomon pilot results

Written 2026-11-11. Pre-registration and all deviations: `PREREG.md`. Raw numbers: `results/pilot_metrics*.json`. The pilot is a feasibility and calibration run (200 MT-Bench items). It is not evidence for a headline claim.

## TL;DR
- **The Rashomon set exists.** At eps = 0.05, 5 of 6 eligible judge configs agree with humans about as well as the best one (kappa 0.49-0.54, paired SE about 0.05). H1 holds.
- **Those configs disagree a lot on individual items, but not on system rankings.**
  - Item level: in 37.6% of items, at least one config in the set gives a different verdict from the best config. Re-running the same config changes only 9-19% of verdicts.
  - System level: their model rankings never flip. Flip rate is 0.0 on the primary path, Kendall tau is 1.0, and the permutation p is 1.0. H2 is not supported.
- **Pre-registered reading:** "validated judges disagree no more than chance on system conclusions", *on MT-Bench's 6 widely separated 2023 systems*.
- **Why that caveat matters:** the 6 systems are far apart, so ranking stability here is the easy case. The real test is close systems, which means Arena human votes. Recommendation: reframe and scale to Arena under a second pre-registration (see the last section).

## Setup (as run)
- **Items.** 200 MT-Bench turn-1 items, 13-14 per model pair.
- **Primary path.** Complete-case, non-CLI configs only (A5/A6). Only items every included config judged successfully are kept: n = 181.
- **Configs.** 12 in total: 5 free judges x pairwise and pointwise, plus Claude Haiku and Sonnet pairwise via the CLI.
- **Eligibility.** A config needs coverage of at least 90% and failures of at most 5%.

| Config | Coverage | Failures (after one transient retry) | Primary path |
|---|---|---|---|
| deepseek-pair | 100% | 3.0% | included |
| deepseek-point | 100% | 5.0% | included |
| gemma-pair | 100% | 0.5% | included |
| gemma-point | 100% | 0.0% | included |
| nemotron-pair | 100% | 3.5% | included |
| nemotron-point | 100% | 0.0% | included |
| mistral-pair | 100% | 11.5% | excluded (failure cap) |
| mistral-point | 100% | 6.0% | excluded (failure cap) |
| gptoss-pair / -point | pending at analysis time | — | see addendum |
| haiku-pair, sonnet-pair | 100% | 0% | sensitivity path only |

The transient retry (deviation 2026-11-10b) recovered 2 mistral-point failures. On the pilot configs, every failure was an API failure (timeout, 429 or 5xx), not a parse failure.

## Agreement with humans
**Human ceiling.** Expert-expert pairwise agreement on the 485 items with more than one vote is **0.641**. This matches the 63-67% "with ties" agreement in Zheng et al. 2023.

| Config | kappa | Tie rate | Paired SE vs best | In-set frequency (bootstrap) |
|---|---|---|---|---|
| nemotron-pair (best) | 0.538 | 11.6% | — | 0.83 |
| gemma-point | 0.525 | 31.5% | 0.055 | 0.69 |
| nemotron-point | 0.512 | 24.9% | 0.055 | 0.54 |
| deepseek-pair | 0.501 | 8.8% | 0.043 | 0.41 |
| gemma-pair | 0.491 | 6.6% | 0.048 | 0.34 |
| deepseek-point | 0.467 | 21.0% | 0.052 | 0.14 |

- **Kappa gaps are inside noise.** Every gap to the best config is within about 1.3 paired SE. The best config itself lands in the set in only 83% of bootstraps. As predicted analytically (A2), membership of the set at n ≈ 200 is mostly noise.
- **Format shows up in tie rates.** Pointwise configs tie 21-31% of the time; pairwise configs tie 7-12%.

## Multiplicity (primary path, eps = 0.05, R = 5 configs)
| Metric | Value |
|---|---|
| Ranking flip rate inside R | 0.000 (bootstrap SE 0.026) |
| Confident flips inside R | 0.000 |
| Flip rate across all 6 configs (A9 contrast) | 0.022 |
| Confident flips across all 6 configs | 0.000 |
| Mean Kendall tau inside R | 1.00 |
| Permutation null mean flip rate (A3) | 0.004; p = 1.0 |
| Item-level ambiguity (Marx) | 0.376 |
| Noise floor: nemotron-pair re-runs | 0 flips, 9.4% of verdicts change |
| Noise floor: nemotron-point re-runs | 0 flips, 19.0% of verdicts change |

- **Eps curve.** The flip rate is 0 for every eps up to 0.07. From eps = 0.08 deepseek-point enters the set, and the flip rate becomes 0.022. All of those flips are in contested model pairs, and none are in clear pairs. There are 6 clear and 9 contested pairs.
- **Split-half (A1 dry run).** Flips measured on held-out items are 0.0 in both folds.
  - The selected best config changes between folds: nemotron-pair in one, gemma-pair in the other.
  - In fold 2, the best config's kappa falls from 0.58 on the selection half to 0.42 on the measurement half. That is the winner's curse A1 was designed for.

### Sensitivity paths
| Path | n | Set size | Flip rate inside R | Flips across all configs | Ambiguity |
|---|---|---|---|---|---|
| Primary | 181 | 5 | 0.000 | 0.022 | 0.376 |
| With Claude CLI (A6) | 181 | 7 | 0.000 | 0.017 | 0.398 |
| Failures as ties (A5) | 200 | 4 | 0.000 | 0.022 | 0.390 |

- **Claude CLI.** Haiku (kappa 0.528) and Sonnet (0.511) both land in the set. Adding them changes nothing at the system level.
- **CLI split-half.** One fold measured a 0.036 flip rate on held-out items. The other fold's set had only 1 member, so there was nothing to measure.

## Hypotheses (pre-registered)
- **H1 (set size of at least 2): holds.** The set has 5 configs (7 with Claude).
- **H2 (flips above noise / permutation): not supported.** Both flip rates are 0.
- **H3 (item ambiguity of at least 10%): holds.** Ambiguity is 37.6%, and it also exceeds the 9-19% that same-config re-runs produce.
- **H4 (format beats model, exploratory): not testable at the system level** with 0 flips. At the item level, format mainly moves the tie rate.

**Reading.** The item-level Rashomon effect is real and larger than re-run noise. But win-rate aggregation over about 13 items per model pair absorbs it, because MT-Bench's systems are far apart. Under the pre-registered decision rule, FR does not exceed noise, so the answer is **reframe, not scale as-is**.

## gpt-oss addendum (2026-11-11)
- **Completion.** gpt-oss-20b finished before the 09:00 cutoff. It went through the same single transient retry: pair 33 retried with 11 still failing, point 23 retried with 6 still failing.
- **Eligibility.**
  - gptoss-pair is **excluded**: 6.5% failures, over the 5% cap.
  - gptoss-point is **eligible**: 3.5% failures. Its kappa of 0.426 is the lowest of any config, with in-set frequency 0.045, so it is **not in R**.
- **Changes to the analysis.** It only changes the complete-case item set, which drops to n = 174. On those items:
  - R = {nemotron-pair, nemotron-point, gemma-point, deepseek-pair}, so gemma-pair drops out.
  - The flip rate inside R is **0.033**, about 3 of 90 comparisons, with SE 0.030. The permutation p is 0.002. Mean tau is 0.93.
  - Confident flips are still **0**. All flips are in contested pairs.
  - Both split-half folds measure 0.044 on held-out items.
- **Why this counts against confidence, not for it.** Adding a judge that never enters the set moved the flip rate from 0 to 3.3% and the permutation p from 1.0 to 0.002. It did so only by removing 7 items. That is exactly the instability the full run needs to be powered against, and a reason not to quote any pilot point estimate.
- **No pre-registered result changes.** H1-H4 read the same.

## Perf audit
- **Analysis.** All analysis paths run in about 10.6 s. They were left unchanged so the numbers stay identical.
- **Cache.** The cache is a direct path per key and skips finished work up front.
- **Judge calls.** Calls already use a thread pool with backoff.
- **The real bottleneck was the provider.** NIM gpt-oss took about 195 s per call against a 180 s client timeout. Raising the timeout to 600 s with 12 workers (deviation 2026-11-10d) took throughput from about 0.5 to about 1.3 calls/min.
- **Lesson for the full run.** Budget wall-clock by the slowest free provider, and drop any judge whose latency would block the run.

## Background claims (verified; sources in `judge-rashomon-research/05-background-claims.md`)
| Claim | Source |
|---|---|
| GPT-4 judge vs human: 85% agreement without ties (human-human 81-82%); 66% with ties (63-67%) | Zheng et al. 2023, arXiv 2306.05685 |
| AlpacaEval length-controlled: Spearman 0.98 with Chatbot Arena | arXiv 2404.04475 |
| Arena-Hard: 90.9% agreement / 0.932 Spearman vs overall Arena (98.6% is against Hard Prompts) | arXiv 2406.11939 |
| Changing the judge changes the ranking (prior art we must cite, not claim) | 2611.04090, 2609.33044, 2610.38076, 2412.09569, 2501.00560 |
| Judge use is widespread: 53.3% of respondents who run evals use LLM-as-judge (n=1,340) | LangChain State of AI Agents survey |

## Caveats
- **Small samples.** There are about 13 items per model pair, only 6 systems and 15 pairs. Every flip estimate has an SE comparable to the estimate itself.
- **Old systems and labels.** The systems date from 2023 and are well separated. The expert labels are often from a single annotator.
- **Position bias untested.** The pilot shows each item in one order, so position bias is confounded with judge disagreement. Fixing this is planned for the full run (A8).
- **Free-tier providers.** Two of five judges (mistral, gpt-oss pairwise) failed eligibility because of the provider, not the judge. Coverage itself is a free-tier selection effect.
- **Claude CLI configs** run at default temperature. They have no noise floor of their own.
- **One look at incomplete data.** One interim look at incomplete gpt-oss data happened while benchmarking throughput. It is disclosed in deviation 2026-11-10d.

## Recommendation: reframe to Arena, under a second pre-registration
MT-Bench answered the feasibility questions. The set is non-trivial, the item-level disagreement is real, and the pipeline works on free infrastructure. It cannot answer the system-level question, because its systems are too far apart.

**Design of the full run** (from `judge-rashomon-research/06-arena-redteam.md`):
- **Data:** pooled LMArena human-preference votes (55k / 100k / 140k; 140k is CC-BY-4.0).
- **Primary analysis:** flip probability as a continuous function of the human vote margin. The clear/contested/unresolved split is descriptive only.
- **Self-preference control:** judge-family self-preference is excluded per comparison.
- **Selection:** the set is selected on the 200 pilot items, and flips are measured on held-out items.
- **Position bias:** a balanced order coin, plus a 40-item subset judged in both orders.
- **Budget:** about 70k free-tier calls.

**Before any Arena call:**
1. Freeze PREREG-arena.md.
2. Re-check the free model IDs.
3. Re-run the novelty search (pre-publish checklist).
