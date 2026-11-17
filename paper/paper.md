# Equally Valid, Differently Ranked? Predictive Multiplicity of LLM Judges at the System Level

*Draft, 2026-11-16. Arena sections filled from `RESULTS-arena.md` (final analysis under deviation 2026-11-16i).*

## Abstract
LLM-as-a-judge evaluations are usually validated by checking that the judge agrees with human labels, and the judge's rankings are then trusted. We ask whether that validation is enough. Among judge configurations that agree with humans equally well (an epsilon-Rashomon set, defined with a standard error), how often do two of them disagree about which of two systems is better? Does that happen more than chance and re-run noise would produce? On 3,200 LMArena battles, the set held a single config (pairwise Nemotron), so within-set flips could not be tested. Configs that a looser agreement threshold would accept disagree on about 22% of system-pair conclusions (95% CI 15–30%), across every sensitivity path. Closeness (H3) could not be tested. We release `judge-rashomon`, a tool that runs this check on any judge setup.

## 1. Introduction
- **Practice.** Validate a judge against a few hundred human labels (kappa, accuracy or correlation), then use it to rank systems and pick winners. Examples: Arena-Hard, AlpacaEval, and many internal leaderboards.
- **Gap.** Agreeing with humans on average does not mean the judge's conclusions about systems are stable. Marx et al. showed that equally accurate models disagree on individual predictions. We move the unit of analysis to system-level conclusions.
- **Question.** Among judges that pass validation equally well, do system-level conclusions flip beyond chance? Do the flips concentrate where humans find two systems close?
- **Contributions.**
  1. A selection and measurement procedure with an exchangeable-config null and a re-run noise floor.
  2. A pre-registered study on MT-Bench (pilot) and LMArena human votes (main).
  3. The `judge-rashomon` tool.

## 2. Related work
See `paper/related_work.md`; it will be condensed here.

## 3. Method
### 3.1 Setup
- **Data.** Items are battles (prompt, answer from system A, answer from system B) with a human preference.
- **Judge configs.** A config is a (judge model, format) pair. There are two formats:
  - **Pairwise:** A, B or tie.
  - **Pointwise:** each answer scored 1-10, with the higher score winning.
- **Verdict mapping.** Every verdict is mapped to the same sorted orientation of the system pair.

### 3.2 Selection: the Rashomon set
- **Split.** Within each system pair, battles are split into a selection half and a measurement half.
- **Agreement.** On the selection half, each eligible config gets Cohen's kappa against human votes, complete-case. Eligible means coverage of at least 90% and failures of at most 5%.
- **The set.** R = the best config, plus every config whose kappa gap to the best is at most one cluster-jackknife SE of that gap (clustered over system pairs).

### 3.3 Measurement: conclusions and flips
- **Conclusion.** On the measurement half, a config's conclusion for a system pair is sign(judge win rate - 0.5), with judge ties counted as 0.5.
- **Flip.** Two configs in R give opposite conclusions on a pair.
- **Self-preference rule.** A config pair is scored on a system pair only if neither judge's model family is one of the two contestants.

### 3.4 Inference
- **Null.** A joint verdict permutation: on each measurement item, the verdicts of R's configs are shuffled among themselves. R is fixed, because the selection half is untouched.
  - **Why joint.** Pairwise swaps break the correlation between config pairs that share a config. In our synthetic calibration they were anti-conservative, rejecting about 20% of the time at a nominal 5%.
- **Noise floor.** The best pairwise and best pointwise configs are re-run twice with a cache-busting nonce. Their flip rate against their own re-runs gives an upper 95% bound.
- **Closeness.** A logistic regression of flip on the human margin |p_h - 0.5|, tested against the null's slope distribution. Under the null, flips also fall with margin, from sampling alone.
- **Uncertainty.** A cluster bootstrap with 2,000 draws:
  - system pairs are resampled, then battles within each half
  - the human margin is redrawn from Binomial(n_votes, p_h)
  - selection is re-run in every draw
- **Calibration on synthetic data.** Known-answer data built on the real items: 40 runs per scenario.

| Scenario | H2 rejections | H3 rejections |
|---|---|---|
| Null | 0% (permutation test alone: 2.7%) | 2.5% |
| Planted disagreement | 57.5% (70% of the permutation tests when the set has at least 2 configs) | 37.5% |

### 3.5 Hypotheses (pre-registered)
- **H1:** |R| is at least 2.
- **H2 (primary):** the flip rate inside R has a permutation p below 0.05 and exceeds the re-run floor.
- **H3:** the slope of flips on margin is more negative than the null's.
- **H4:** the contrast between all configs and R.
- **H5 (exploratory):** whether format or judge model explains more flips.

## 4. Pilot: MT-Bench (feasibility)
- **Setup.** 200 MT-Bench turn-1 items, 6 systems from 2023, expert labels. 12 configs: 5 free judges × 2 formats, plus Claude Haiku and Sonnet, pairwise, as a sensitivity path.
- **Agreement.**
  - 5 of 6 eligible configs fall within about 1 SE of the best (kappa 0.49-0.54).
  - Expert-expert agreement is 0.641, which matches Zheng et al.
- **Item level.** At least one config in the set disagrees with the best config on 37.6% of items. Re-runs of the same config change 9-19% of verdicts.
- **System level.** Zero ranking flips inside the set. Kendall tau is 1.0, and the permutation p is 1.0.
- **Reading.** MT-Bench's systems are far apart, so it cannot test the question. This motivated the reframing toward close pairs.
- **Fragility.** One judge that never entered the set moved the flip rate from 0 to 3.3%, through complete-case item loss alone. That is why we report contrasts with nulls, not point estimates.

## 5. Main study: LMArena
- **Data.** Pooled open LMArena releases (55k / 100k / 140k).
  - **Filters:** English, single-turn, prompt of at least 5 words, responses of at most 4,000 words, de-duplicated.
  - **Pairs:** 80 system pairs with at least 100 decisive votes.
  - **Battles:** 40 per pair, split 20/20, giving 3,200 items. A 320-item subset is judged in both orders.
- **Judges.** gpt-4o-mini (Batch API), and Nemotron-3-Ultra, DeepSeek-v4.1-flash and Mistral-large-4 (NIM), each pairwise and pointwise. Four configs were eligible: both gpt and both Nemotron formats. Mistral and DeepSeek failed the failure cap or coverage rule because the NIM runs were parked before retries (deviation i).
- **Deviations.** All are listed in PREREG-arena (a–i). None were made after looking at a measurement-half verdict. The last one parked the NIM runs, so there is no Arena noise floor.
- **5.1 Set and agreement** (fig: kappa.png). Kappa: nemotron-pair 0.354, gpt-pair 0.295, nemotron-point 0.195, gpt-point 0.156. gpt-pair trails by 2.2 gap SEs, so R = {nemotron-pair}; H1 is not supported. gpt-pair joins R in 26.5% of bootstrap draws.
- **5.2 Flips vs chance and noise (H2).** Not testable with |R| = 1. In bootstrap draws where R has two members, the within-set flip rate is 8–34% (descriptive only).
- **5.3 Closeness (H3).** Not testable with |R| = 1.
- **5.4 Set vs all (H4), and format vs model (H5).** Across all four eligible configs, 22.0% of system-pair conclusions flip (CI 14.9–30.2%). Pairwise beats pointwise by 0.10–0.20 kappa for both models; format separates configs more than model does.
- **5.5 Sensitivity multiverse** (fig: multiverse.png). R stays {nemotron-pair} on every path. The all-config flip rate stays 19.6–22.0%. Fixed epsilon ≥ 0.06 admits gpt-pair, and the two then disagree on 18.9% of comparisons. Agreement when the presentation order is flipped ranges from 79% to 92% at the item level.

## 6. Limitations
- **Few judges.** Only 3-4 judge models are available on free or cheap infrastructure, and two have family contestants. Format and model are only partly separable.
- **Moderate power.** On synthetic data, H2 power is about 0.6 and H3 power about 0.4. A null result is not evidence of no multiplicity.
- **Crowd votes reward style.** Kappa against them may select for style alignment; the length covariate checks this.
- **Older contestants.** Public Arena releases skew toward older systems.
- **Provider drift.** Free-tier serving changes over time, so model IDs and providers are logged on every call.

## 7. Conclusion
On Arena, a 1-SE agreement rule was strict enough to leave one judge standing, so the pre-registered multiplicity question has no confirmatory answer here. Judges that a looser rule would accept disagree on about a fifth of system-level conclusions. Whether this exceeds re-run noise is untested, because the noise floor was parked. The practical reading: the threshold for "validated" decides how much ranking instability a team inherits.

## Artifacts
- **Code and tool:** `judge_rashomon/` (CLI, figures, known-answer tests).
- **Pre-registrations:** `PREREG.md` (pilot) and `PREREG-arena.md` (main, with the deviation log).
- **Verdict caches:** released subject to the dataset licenses.
