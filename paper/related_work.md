# Related work (draft, 2026-11-11)

Every arXiv ID below was checked against the arXiv API on 2026-11-11. Each line is a claim we make about the paper, so the claims need re-reading against the full text before posting (pre-publish checklist).

## Predictive multiplicity and Rashomon sets
- **Marx, Calmon & Ustun (2019/2020), Predictive Multiplicity in Classification (1909.06677).** Models that are nearly equally accurate can disagree on individual predictions. They define ambiguity and discrepancy.
  - **We borrow:** the epsilon-set idea and item-level ambiguity.
  - **We move the unit:** from individual predictions to system-level conclusions ("A beats B").
- **Semenova, Rudin & Parr (2019), On the Existence of Simpler Machine Learning Models (1908.01755).** The size of the Rashomon set, and why a large set implies that simple, accurate models exist.
- **Chou et al. (2026), Adversarial Multiplicity (2611.07378).** How multiplicity can be exploited to steer conclusions. They recommend pre-registration and sensitivity analyses, which we follow.

## Disagreement among metrics (the closest ancestor)
- **Mathur et al. (2020), Tangled up in BLEU (2006.06264).** Machine-translation metrics with similar correlation to humans disagree on system rankings, especially for close systems.
- **Deutsch et al. (2022), Re-Examining System-Level Correlations of Automatic Summarization Evaluation Metrics (2204.10216).** Correlation at the system level hides instability on close pairs.
- **How we differ:** both papers compare metrics by their correlation with humans. We condition on a statistically defined equal-agreement set, and test disagreement against an exchangeable-config null and a re-run noise floor.

## LLM-as-a-judge agreement and ranking
- **Zheng et al. (2023), MT-Bench and Chatbot Arena (2306.05685).** GPT-4 as a judge agrees with humans about as well as humans agree with each other. Our pilot dataset and human ceiling come from here.
- **Chiang et al. (2024), Chatbot Arena (2403.04132).** The crowd human-preference data our full run uses.
- **Dubois et al. (2024), Length-Controlled AlpacaEval (2404.04475)**, and **Li et al. (2024), Arena-Hard / BenchBuilder (2406.11939).** Judge-based benchmarks validated by high correlation with Arena rankings. That is exactly the validation step whose sufficiency we test.
- **Gera et al. (2024), JuStRank (2412.09569).** Benchmarks judges by how well they rank systems, and notes that agreement at the label level does not guarantee stable rankings.
- **Gao et al. (2024), Re-evaluating Automatic LLM System Ranking (2501.00560).** Scores each judge setup by its correlation with Arena, and shows that judges fail on close systems. It does not compare equally good setups against each other.

## Instability of judge-based rankings (concurrent, 2026)
- **Han & Kim (2026), Swapping the Judge Changes the Ranking (2611.04090).** A pre-registered study in which swapping the judge provider changes system comparisons.
  - **How we differ:** we ask whether judges that pass human validation *equally well* still disagree, beyond chance and re-run noise. We also report the contrast between the validated set and all judges, which directly qualifies their finding.
- **Ayyagari (2026), Pinned and Still Unstable (2609.33044).** Verdict variance within a single judge, and the noise floor of judge leaderboards. Our re-run floor plays the same role, and must be beaten for H2 to hold.
- **Subasic et al. (2026), Prompt Lottery in LLM-as-a-Judge (2610.38076).** Rephrasing the prompt shifts verdicts. The prompt axis is fixed in our design and is future work.
- **Lu et al. (2026), Evaluation Contracts / EvalLever (2611.04099).** Rank flips caused by evaluation contracts, including grids of LLM judges. They do not condition on equal agreement with humans.

## Biases that threaten the design (handled in the pre-registration)
- **Panickssery et al. (2024), LLM Evaluators Recognize and Favor Their Own Generations (2404.13076)**, and **Wataoka et al. (2024), Self-Preference Bias in LLM-as-a-Judge (2410.21819).** Handled by our per-comparison family exclusion rule.
- **Singh et al. (2025), The Leaderboard Illusion (2504.20879).** Arena samples battles non-uniformly, which is why our estimand is the head-to-head win rate on each pair's own battles, not Bradley-Terry.

## Our position in one sentence
Prior work shows that judge choice changes rankings. We test whether *passing human validation* protects against that: whether judges that are equally good by the standard validation check still disagree on which system is better, beyond chance and re-run noise, and whether that disagreement concentrates where humans themselves find the systems close.
