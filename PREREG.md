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
