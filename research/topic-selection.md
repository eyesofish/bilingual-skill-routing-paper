# Topic Selection

## Selected topic

**Bilingual Skill Routing for Personal Agent Registries: Metadata, Sparse
Retrieval, Abstention, and Latency**

The paper studies how a local agent should select one reusable `SKILL.md` from
a small but overlapping personal registry when requests are written in
English, Chinese, or both languages.

## Research question

For personal registries with tens of skills, how much do bilingual metadata
and character-aware sparse retrieval improve skill selection and abstention,
and what latency do they add on a laptop CPU?

## Candidate comparison

Scores use a 1-5 scale. Higher is better.

| Candidate | Local evidence | Safe to publish | Reproducible now | Novelty room | Total |
|---|---:|---:|---:|---:|---:|
| Bilingual skill routing | 5 | 5 | 5 | 4 | 19 |
| Long-session agent recovery | 5 | 1 | 2 | 4 | 12 |
| Job-application agent evaluation | 5 | 2 | 3 | 3 | 13 |
| Personal recommender-system retrospective | 4 | 4 | 3 | 2 | 13 |
| Multimodal learning survey | 3 | 5 | 2 | 2 | 12 |

## Why this topic won

1. `~/personal/eyesofishTUI` already contains an original, tested skill router
   with explainable ranking, optional reranking, and timeout fallback.
2. `~/personal/bilingual-tech-terms/evals/trigger-evals.json` already contains
   realistic Chinese, English, and near-miss trigger prompts.
3. The safe local corpus contains dozens of `SKILL.md` files, including a
   controlled 16-skill fixture set.
4. Public large-scale work such as SkillRouter studies full-text routing over
   about 80K skills, but its public paper does not report a dedicated Chinese
   or code-switched evaluation. This project therefore focuses on a narrower
   multilingual, small-registry, CPU-only setting.
5. The project can report real retrieval and latency measurements without
   remote APIs, private datasets, or unsupported claims.

## Contribution hypotheses

These are hypotheses, not results.

1. Bilingual aliases will improve Chinese and mixed-language Hit@1 more than
   they improve English Hit@1.
2. Character n-gram retrieval will be more tolerant of Chinese text and
   code-switching than whitespace token overlap.
3. A simple rank fusion will improve overall retrieval but may make abstention
   harder unless its threshold is calibrated.
4. All sparse methods will stay well below an interactive latency budget on a
   personal registry with fewer than 50 skills.

## Local evidence map

| Evidence | Role in this study |
|---|---|
| `~/personal/eyesofishTUI/src/search/rank.ts` | Existing explainable weighted-token baseline |
| `~/personal/eyesofishTUI/src/search/pipeline.ts` | Existing timeout and fallback design |
| `~/personal/eyesofishTUI/test/fixtures/skills/` | Controlled skill registry |
| `~/personal/bilingual-tech-terms/evals/trigger-evals.json` | Realistic bilingual and near-miss query patterns |
| `~/.copilot/skills/*/SKILL.md` | Personal skill-registry metadata |
| `~/personal/skill-router-tui-research/` | Earlier problem framing; leaked-source references are excluded |

