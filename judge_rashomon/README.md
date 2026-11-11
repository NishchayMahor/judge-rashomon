# judge-rashomon

Your LLM judge agrees with human labels. Would another judge that agrees with humans just as well rank your systems the same way?

`judge-rashomon` takes verdicts from several judge configurations plus human labels and reports:
- **the Rashomon set:** configs whose agreement with humans (Cohen's kappa) is within 1 standard error of the best, chosen on a selection split
- **flip rate:** how often two configs in the set reach opposite conclusions about which system is better, on a held-out measurement split
- **a joint permutation null:** whether that flip rate is beyond chance. Verdicts are shuffled among the configs in the set on each item.
- **an optional re-run noise floor:** add rows with `config#rerun1`
- **flip probability against human closeness:** how close the human vote was for each system pair

```bash
judge-rashomon verdicts.csv --families families.json --out report.json
```

**CSV columns:** `item_id, system_1, system_2, human (1|2), config, verdict (1|2|tie|fail)`, plus an optional `split (select|measure)`.

**families.json:** optional, `{config: [contestant name prefixes it must not judge]}`, for the self-preference rule.

**Testing:** the tests use synthetic data with a known answer. The null is calibrated, and a planted disagreement is detected. `examples/` holds one dataset of each kind.
