# Bilingual Skill Routing Paper

Reproducible research workspace for:

> **Bilingual Skill Routing for Personal Agent Registries: Metadata, Sparse
> Retrieval, Abstention, and Latency**

The study evaluates deterministic routing from English, Chinese, and
code-switched requests to a registry of reusable `SKILL.md` instructions. It
uses a sanitized local catalog, no remote model APIs, and no private work data.

## Current research question

For personal registries with tens of overlapping skills, how much do bilingual
metadata and character-aware sparse retrieval improve skill selection and
abstention, and what latency do they add on a laptop CPU?

## Reproduce

```bash
python3 scripts/build_catalog.py
python3 scripts/evaluate.py
python3 -m unittest discover -s tests -v
```

Generated artifacts:

- `data/catalog.json`: sanitized metadata snapshot.
- `results/baseline.json`: machine-readable metrics and errors.
- `results/baseline.md`: compact result table.

## Benchmark design

- Two metadata conditions: original and bilingual alias augmentation.
- Three deterministic methods: weighted token overlap, character n-gram
  TF-IDF cosine, and reciprocal-rank fusion.
- Development split for abstention-threshold selection.
- Test metrics: Hit@1, Hit@3, MRR, abstention accuracy, combined routing
  accuracy, and per-query CPU latency.
- Query strata: English, Chinese, mixed language, and no-skill requests.

## Repository map

- `research/topic-selection.md`: candidate scoring and evidence map.
- `research/privacy-boundary.md`: publishable and excluded evidence.
- `config/sources.json`: explicit source allowlist.
- `data/queries.json`: human-authored bilingual benchmark.
- `data/metadata_augments.json`: bilingual metadata treatment.
- `scripts/`: catalog construction and evaluation.
- `paper/`: outline, claims, related work, and experiment plan.
- `results/`: generated baseline measurements.

## Scope boundary

This repository does not use leaked source code, Copilot internal details,
private chat or email exports, work logs, credentials, or personal application
records. It does not claim end-to-end agent task success.
