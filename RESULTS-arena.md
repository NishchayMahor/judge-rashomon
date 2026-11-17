# Results: LMArena main study (final, 2026-11-16)

Pre-registration: `PREREG-arena.md`. This analysis runs the frozen pipeline unchanged, with the deviations listed there. Deviation 2026-11-16i matters most: the NIM runs were parked before they finished, so DeepSeek pointwise is incomplete, most transient failures never got their one retry, and the noise floor was not run.

Reproduce: `uv run python src/arena_release2.py --out results/arena_final` (2,000 bootstrap draws, 1,000 permutations; about 2 minutes from the verdict cache).

## Eligibility

| Config | Coverage | Failures | Eligible |
|---|---|---|---|
| gpt-pair | 1.000 | 0.000 | yes |
| gpt-point | 1.000 | 0.000 | yes |
| nemotron-pair | 1.000 | 0.006 | yes |
| nemotron-point | 1.000 | 0.001 | yes |
| mistral-pair | 1.000 | 0.126 | no (failures) |
| mistral-point | 1.000 | 0.084 | no (failures) |
| deepseek-pair | 1.000 | 0.227 | no (failures) |
| deepseek-point | 0.725 | 0.213 | no (incomplete) |

Most Mistral and DeepSeek failures are NIM timeouts and server errors. Under the pre-registered one-retry rule, some of these configs could have become eligible. That retry was parked (2026-11-16i), so their exclusion reflects provider throughput, not judge quality.

## Primary path

**Agreement with humans (selection half, Cohen's kappa):**

| Config | Kappa | Gap SE to best | Share of bootstrap draws in R |
|---|---|---|---|
| nemotron-pair | 0.354 | best | 99.7% |
| gpt-pair | 0.295 | 0.027 | 26.5% |
| nemotron-point | 0.195 | 0.021 | 0% |
| gpt-point | 0.156 | 0.023 | 0% |

**The Rashomon set:**
- **R = {nemotron-pair}.** gpt-pair is 0.059 behind, about 2.2 gap SEs, outside the 1-SE rule.
- **H1 (|R| ≥ 2): not supported.** With one config in R there are no within-set flips, so H2 and H3 cannot be tested on the point estimate.
- **Bootstrap.** Across draws, R has two members about a quarter of the time (gpt-pair joining). In those draws the flip rate inside R has a 95% interval of 8.1% to 34.4%. This is descriptive, not a test.

**Across all eligible configs (H4, descriptive).**
- Two eligible configs reach opposite conclusions on 22.0% of scored system-pair comparisons (95% CI 14.9% to 30.2%).
- The all-minus-R contrast is undefined on the point estimate. Its bootstrap interval is −8.0 to +12.0 points.

**Format effect.** Pairwise configs beat pointwise ones by 0.10 to 0.20 in kappa for both judge models. The validation step separates formats clearly. Within the pairwise format, the two judge models were not separated with confidence: gpt-pair is in R in a quarter of draws.

## Sensitivity paths (exploratory)

| Path | R | Flip rate across all eligible configs |
|---|---|---|
| Primary | {nemotron-pair} | 22.0% |
| 10-word prompts | {nemotron-pair} | 22.0% |
| No family exclusion | {nemotron-pair} | 19.6% |
| Ties dropped | {nemotron-pair} | 22.0% |
| Bradley-Terry conclusions | {nemotron-pair} | 19.7% |

- **Epsilon grid.**
  - With a fixed epsilon of 0.06 or more, R gains gpt-pair. The two then disagree on 18.9% of comparisons.
  - At epsilon 0.16, R has 3 configs and a 22.4% flip rate.
  - At epsilon 0.20, R has 4 configs and a 22.0% flip rate.
- **Order (both-orders subset, item-level agreement when the presentation is flipped):**
  - gpt-pair 79.1%
  - nemotron-pair 82.5%
  - mistral-pair 84.4%
  - deepseek-pair 92.0%
- **Length model and flip classes:** not applicable, because |R| < 2.

## Reading

1. **The 1-SE validation rule separated the judges.** On 3,200 Arena battles, only one config passed as "as good as the best". The pre-registered question, whether equally valid judges flip conclusions, therefore has no confirmatory answer here. The set it needs did not form.
2. **Looser validation would leave conclusions unstable.** Configs that a looser rule would accept disagree on about a fifth of system-pair conclusions. That holds at fixed epsilons from 0.06 to 0.20 and across every sensitivity path. So what a judge says about which system is better depends heavily on which validated-looking judge you pick, unless the agreement check is tight enough to rule most of them out.
3. **Format mattered more than model** for agreement with humans, in this judge pool.

## What this does not show

- **No noise floor.** Re-run variation was never measured on Arena, so the 22% is not compared against what a single judge would do on a second run. The pilot's item-level re-run disagreement was 9–19%, on different data.
- **Pool too small.** Four eligible configs from two judge models. Mistral and DeepSeek fell out for throughput reasons.
- **Not confirmatory for H2 or H3.** The study ended under a deviation that parked the noise floor and most retries.
