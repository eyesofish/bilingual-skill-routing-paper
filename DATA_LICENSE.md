# Data License and Attribution

## Multilingual query extension

`data/skillrouter_queries_multilingual.jsonl` is a derivative of the public
SkillRouter Eval Core task file:

- Dataset: <https://huggingface.co/datasets/pipizhao/SkillRouter-Eval-Core>
- Paper: <https://arxiv.org/abs/2603.22455>
- Upstream task source: <https://github.com/benchflow-ai/skillsbench>

The original SkillsBench repository is licensed under Apache License 2.0. The
English task text remains unchanged. The Simplified Chinese and code-switched
variants are modifications and are distributed under Apache License 2.0. Users
must retain this attribution and the upstream notices.

The large candidate pool, relevance labels, and model caches are not committed.
The download script retrieves them from the public SkillRouter release and
verifies their SHA-256 hashes.

## Personal-registry pilot

The small catalog snapshot and query set under `data/catalog.json` and
`data/queries.json` are provided for research reproduction. Skill descriptions
retain the licenses and attribution of their original skill packages. Do not
assume that this repository relicenses third-party skill text.

