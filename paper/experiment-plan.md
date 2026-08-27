# Experiment Plan

## Current completed experiment

### Catalog

- 26 skills.
- 10 allowlisted user skills and 16 controlled `eyesofishTUI` fixtures.
- Only `name`, `description`, `when_to_use`, `aliases`, and `tags` are used.
- Stored paths are sanitized source labels, not absolute local paths.

### Queries

- 51 total prompts.
- 17 development prompts and 34 test prompts.
- 42 positive prompts across 14 target skills.
- 9 no-skill prompts for abstention.
- English, Chinese, and mixed-language prompts appear in both splits.

### Treatments

1. **Original metadata:** metadata as found in each allowed `SKILL.md`.
2. **Bilingual augmented:** original metadata plus a small human-written alias
   list for the target skills.

### Methods

1. **Weighted token:** explainable exact, substring, and field-weighted token
   overlap based on the existing `eyesofishTUI` baseline.
2. **Character n-gram:** TF-IDF cosine over 2-, 3-, and 4-character n-grams.
3. **RRF fusion:** reciprocal-rank fusion of weighted-token and character
   rankings.

### Threshold rule

For each method and metadata condition, choose the activation threshold that
maximizes routing accuracy on the development split. Break ties using
development abstention accuracy. Never tune on the test split.

### Metrics

- Hit@1 and Hit@3 on positive test prompts.
- Mean reciprocal rank on positive test prompts.
- Abstention accuracy on negative test prompts.
- Combined routing accuracy on all test prompts.
- Hit@1 by language.
- Median, p95, and maximum per-query CPU latency.

## Initial findings to verify

1. Bilingual metadata improves both token and character methods.
2. Mixed-language prompts benefit more than Chinese-only prompts in the current
   sample.
3. RRF improves ranking but has an unusable abstention score distribution.
4. Long Chinese descriptions create broad token overlap and false positives.

## Next experiment queue

### E1. Better Chinese tokenization

Compare single-character tokens with word segmentation and character bigrams.
Measure whether this reduces long-description bias.

### E2. Abstention features

Compare absolute score, top-1 versus top-2 margin, normalized score, and a
small logistic calibration model. Use repeated stratified folds instead of one
development split.

### E3. Metadata ablation

Evaluate name only, name plus description, all frontmatter, bilingual aliases,
and short bilingual trigger summaries.

### E4. Public corpus expansion

Expand from 26 to at least 100 license-audited public skills. Deduplicate
near-identical skills and report overlap groups.

### E5. Independent annotation

Ask at least one second annotator to write or review queries and gold labels.
Report agreement and adjudication rules.

### E6. Stronger retrievers

Add BM25, a multilingual embedding model, and a compact reranker. Report model
download size, warm/cold latency, memory use, and accuracy.

### E7. Statistical analysis

Use bootstrap confidence intervals and paired tests over query-level outcomes.
Avoid broad conclusions until the confidence intervals and corpus expansion are
complete.

## Reproduction commands

```bash
python3 scripts/build_catalog.py
python3 scripts/evaluate.py
python3 -m unittest discover -s tests -v
```

