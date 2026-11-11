# Pre-registration: judge-rashomon pilot

Frozen 2026-11-10, before any judge call was made. Changes after this date go in the Deviations section at the bottom, with a date and a reason.

## Question
Take a set of LLM-judge configurations that agree with human labels about equally well, meaning within epsilon of the best one. How often do they disagree on which of two models is better?

## Data
- `lmsys/mt_bench_human_judgments`, `human` split, turn 1 only.
- If several experts voted on the same (question, model pair), the label is their majority, and a tied vote counts as `tie`. That leaves 910 unique items.
- Pilot sample: 200 items, round-robin across the 15 model pairs (13 or 14 each), seed 0. Saved in `data/pilot_items.json`.
- Six candidate systems: alpaca-13b, claude-v1, gpt-3.5-turbo, gpt-4, llama-13b, vicuna-13b-v1.2.

## Judge configurations (pilot: 10)
5 judge models on the NVIDIA NIM free tier x 2 formats:
- **pairwise**: an adapted MT-Bench pairwise prompt that returns `[[A]]`, `[[B]]` or `[[C]]` (tie). Each item is shown in one order only. The order is fixed per item by a seeded coin (`swap` field), and every config sees the same order.
- **pointwise**: each answer gets its own 1-10 score from an MT-Bench single-answer prompt. The verdict is the higher score, and equal scores count as a tie.

Decoding: temperature 0, max_tokens 1024. Model ids, request dates and raw outputs are cached in `cache/`. The prompt axis (several templates) is left for the full run. In the pilot, the only things that vary are model and format.

## Metrics
1. **Agreement with humans**: Cohen's kappa over 3 classes (A, B, tie) on the 200 items. Parse failures count as ties and are reported separately.
2. **Human ceiling**: on items with 2 or more expert votes, the pairwise agreement among the experts.
3. **System score** per config: each model's win rate across its items, with a tie worth 0.5. Model ranking is by system score.
4. **epsilon-Rashomon set** R(eps) = {configs with kappa >= kappa_max - eps}.
5. **Pairwise ranking flip rate** FR(eps): over every pair of configs in R(eps) and all 15 model pairs, the fraction where the two configs order the model pair oppositely. Equal win rates count as no order, and they are excluded from that comparison.
6. **Kendall tau-b** between the system rankings of each pair of configs in R(eps).
7. **Item-level ambiguity** (Marx et al. 2020): the fraction of items where at least one config in R(eps) gives a different verdict from the best config.
8. **Noise floor**: 2 configs, the best pairwise one and the best pointwise one, each re-run 2 more times with identical requests and a cache-busting nonce. We compute FR and ambiguity between re-runs of the same config.
9. Uncertainty: a 1,000-sample bootstrap over items. Models' system scores are recomputed on each resample.

## Thresholds
- eps in {0.02, 0.05, 0.10}. Primary: **eps = 0.05**.

## Hypotheses (the pilot is underpowered; these are directional checks)
- **H1**: at eps = 0.05, R(eps) holds at least 2 configs. If it holds fewer, the project needs more configs before the question can be asked.
- **H2**: FR(0.05) is above the re-run noise floor FR_noise. The pilot check is whether FR(0.05) exceeds the mean noise FR by more than its bootstrap SE.
- **H3**: item-level ambiguity at eps = 0.05 is at least 10%.
- **H4 (exploratory)**: format (pairwise vs pointwise) changes the ranking more than judge model does.

## Decision rule for scaling to ~50 configs
Scale if H1 holds and FR(0.05) is larger than the noise floor by an amount that the full run (500 items, 50 configs) could plausibly confirm. Stop or reframe if the Rashomon set never holds more than 1 config, or if FR is indistinguishable from noise.

## Known limitations, stated in advance
- Each model pair has about 13 items, so pairwise system orderings are noisy. That is why we bootstrap.
- MT-Bench candidate models are from 2023 and not current.
- The human labels are expert votes from 2023, often from a single annotator.

## Deviations
- **2026-11-10, judge models (before any pilot verdicts were seen).** NIM has no Qwen or current Llama models, and Mistral-large-2 returns 404, so the 5 free judges are: DeepSeek v4.1-flash, Mistral-large-4, Gemma-4-31B and gpt-oss-20b (all on NIM), plus Nemotron-3-ultra via OpenRouter (free tier, pinned to the Nvidia provider with fallbacks off; the serving provider is logged on every call). max_tokens went from 1024 to 4096 so the reasoning models don't get cut off before giving a verdict.
- **2026-11-10, 2 extra configs added (approved by Nishchay, before any pilot verdicts were seen).** "Claude via CLI": claude-haiku-5-5 and claude-sonnet-5-5 through the local `claude -p` CLI on his subscription, pairwise only (400 calls, pilot only). The CLI has no temperature setting, so these run at default sampling. They take part in the Rashomon set like any other config, and results are also reported with them left out. The pilot now has 12 configs.
- **2026-11-10, retries.** A reply with no `[[...]]` verdict is retried up to twice before being cached as a parse failure.
- **2026-11-10, analysis amendment A1-A10 (after a literature pass, a blind red-team and AERS multiplicity guidance; before any pilot metrics were computed on real data).** Full reasoning in `judge-rashomon-research/04-dossier.md`. Partial kappas for 3 configs were seen once, by accident, during a code dry run on an incomplete cache. The amendment does not depend on them: every change comes from the red-team, which never saw any data.
  - A1: select the set on one set of items and measure flips on others. Full run: select on these 200, measure on the 710 unused items. Pilot: 2-fold dry run by question.
  - A2: eps on a dense grid from 0 to 0.20, primary 0.05. Also report paired-kappa SEs, how often each config lands in the set across bootstraps, and the whole curve.
  - A3: primary null for H2 is the verdict-swap permutation (500 draws). The re-run noise floor is kept as a lower bound. H2 is now: permutation p < 0.05 for the mean flip rate inside the set (full run only).
  - A4: the main metric is confident flips (both configs' 95% CIs exclude 0, opposite signs). Results are also split into clear pairs (human gap >= 0.15) and contested pairs.
  - A5: parse failures and API errors count as missing; complete-case is primary. Configs need at least 90% coverage and at most 5% failures. Scoring failures as ties is a sensitivity path.
  - A6: the primary analysis excludes the Claude CLI configs; the with-CLI version is a sensitivity path. Noise-floor configs come from eligible non-CLI configs only.
  - A7: bootstrap resamples whole questions.
  - A8: report tie rates alongside kappa. Full run: balance the order coin within each model pair, and judge a 40-item subset in both orders.
  - A9: headline contrast is flips inside the set vs flips across all configs.
  - A10: system score stays win rate with ties = 0.5. Kendall tau is tau-a over model pairs. These choices are fixed here; alternatives go only into a labelled multiverse figure.
  - Pointwise scores outside 1-10 count as failures.
  - gpt-oss: if it is still running at 2026-11-11 09:00 PT it is stopped. It enters the analysis only if it meets the A5 coverage rule.
  - Null result: if inside-set flips are not above the permutation null, we report "validated judges disagree no more than chance on system conclusions", with the same prominence.
- **2026-11-10b (before any results were computed): one retry pass for transient failures.** Every rep-0 failure was classified from its cached error message. All were transient (timeouts, connection errors, HTTP 429 and 503); none were parse failures. Each transient failure is moved to `cache_failed/` for audit and re-sent once, with the same prompt, parameters and provider pin, through the standard call wrapper. Replies with no verdict stay failures. The 5% failure cap (A5) is applied after the retry. Counts before the retry: deepseek-pair 2/200, deepseek-point 8, gptoss-pair 27 (pairwise still running at the time), mistral-pair 22/200, mistral-point 2. Counts after the retry are logged in `logs/driver.log` and reported in RESULTS-pilot.md. Approved by Nishchay.
