# Bilingual Skill Routing at 80K Scale

Reproducible AAAI-format research workspace for:

> **Bilingual Skill Routing at 80K Scale: Domain Adaptation and Instruction
> Prefixes**

The study evaluates routing from English, Simplified Chinese, and Chinese
requests with a fixed bilingual routing instruction to reusable `SKILL.md`
instructions over a public hard-tier benchmark with 79,141 candidate skills
and 73 tasks.

All experiments run locally. No private work data is included.

## Conference status

The repository uses the official AAAI-27 author kit, the latest official kit
available when this manuscript was prepared. The AAAI-27 full-paper deadline
was July 28, 2026 and has passed. Before submission, update the style and rules
for the next open AAAI cycle. This repository does not claim acceptance.

## Current research question

How robust is metadata-only skill retrieval when English registry entries are
queried in Chinese, how does a fixed bilingual routing instruction change the
result, and how much do domain adaptation and lexical fusion matter?

## Reproduce

```bash
make setup
make data
make validate-data
make cache-e5
make cache-skillrouter
make experiments
make test
make paper
make checklist
make submission
```

Generated artifacts:

- `paper/main.pdf`: anonymous AAAI-format manuscript.
- `paper/reproducibility-checklist.pdf`: completed AAAI checklist.
- `data/skillrouter_queries_multilingual.jsonl`: bilingual public-query
  extension.
- `results/skillrouter_multilingual.json`: large-pool metrics and statistical
  tests.
- `build/aaai-submission.zip`: generated paper, checklist, and source package.
- `build/code-data-supplement.zip`: self-contained anonymous reproducibility
  package.

## Benchmark design

- Public hard tier: word TF-IDF, BM25, character TF-IDF, multilingual-E5,
  SkillRouter-Embedding-0.6B, and reciprocal-rank fusion.
- Public metrics: Hit@1, Hit@5, MRR@20, Recall@10, NDCG@10, bootstrap confidence
  intervals, McNemar tests, indexing time, and query latency.

## Repository map

- `paper/`: AAAI source, PDF, figures, bibliography, and checklist.
- `data/`: multilingual queries and public-data checksums.
- `scripts/`: downloading, validation, embedding, evaluation, and paper-table
  generation.
- `results/`: machine-readable metrics and compact summaries.
- `research/`: internal topic-selection evidence, excluded from anonymous
  submission archives.

## Scope boundary

This repository does not use leaked source code, Copilot internal details,
private chat or email exports, work logs, credentials, or personal application
records. It does not claim end-to-end agent task success or conference
acceptance. See `DATA_LICENSE.md` for upstream dataset attribution.
