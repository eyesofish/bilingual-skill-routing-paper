# Paper Outline

## Working title

**Bilingual Skill Routing for Personal Agent Registries: Metadata, Sparse
Retrieval, Abstention, and Latency**

## 1. Introduction

**Paragraph role: setting.** Agent hosts increasingly expose reusable skills
through a `SKILL.md` package with a name and description that are visible
before the full instructions are loaded.

**Paragraph role: problem.** A personal registry is smaller than a public
marketplace, but its skills often overlap and its user requests may mix Chinese
intent with English technical terms. Wrong routing can load irrelevant
instructions, while always choosing a skill creates false activations.

**Paragraph role: gap.** Large-scale skill-routing work studies retrieval over
tens of thousands of skills. Chinese tool-use benchmarks study tool invocation.
The small-registry, code-switched, metadata-design and abstention setting is not
the main focus of those evaluations.

**Paragraph role: approach.** Build a 26-skill local registry and a 51-query
benchmark with English, Chinese, mixed-language, and no-skill cases. Compare
original metadata with bilingual aliases and three deterministic CPU methods.

**Paragraph role: findings.** Bilingual metadata gives the largest reliable
gain in the first run. Rank fusion gives the best Hit@1 but cannot abstain with
the current score and threshold design.

**Paragraph role: contributions.**

1. A compact bilingual skill-routing benchmark with explicit no-skill cases.
2. A metadata-language ablation over deterministic, explainable retrievers.
3. A separate analysis of retrieval, abstention, and laptop CPU latency.

## 2. Background and Related Work

1. Agent Skills specification and progressive disclosure.
2. Large-scale skill routing and SkillsBench.
3. Tool awareness and selection benchmarks.
4. Chinese tool-use evaluation.
5. Sparse, dense, and hybrid retrieval for routing.

## 3. Task Definition

1. Input: a user request and a registry of skill metadata.
2. Output: a ranked skill list or `NONE`.
3. Query groups: English, Chinese, mixed language, no skill.
4. Metrics: Hit@1, Hit@3, MRR, abstention accuracy, routing accuracy, latency.

## 4. Benchmark

1. Catalog construction from an explicit allowlist.
2. Privacy and publication exclusions.
3. Human-written prompts and development/test split.
4. Original metadata and bilingual alias treatment.
5. Current limitations: one author, small catalog, synthetic balance.

## 5. Methods

1. Explainable weighted token overlap.
2. Character n-gram TF-IDF cosine.
3. Reciprocal-rank fusion.
4. Development-only abstention threshold.
5. Deterministic local evaluation.

## 6. Results

1. Main retrieval and routing table.
2. Results by language.
3. Latency comparison.
4. Error groups:
   - broad Chinese descriptions dominate character/token overlap;
   - paper reading and paper writing remain close;
   - translation and bilingual terminology are confused;
   - rank fusion gives almost constant top scores and weak abstention.

## 7. Discussion

1. Metadata language coverage versus retrieval complexity.
2. Why top-k quality does not imply safe activation.
3. Description length and generic-word bias.
4. Implications for personal skill authors and host runtimes.

## 8. Limitations and Ethics

1. Small, author-written benchmark.
2. No end-to-end task success measurement.
3. No statistical confidence intervals yet.
4. Exclusion of private and work data.
5. Public release requires a final license audit of every catalog item.

## 9. Conclusion

Summarize the initial evidence and state the next evaluation needed before
making a broader claim.

