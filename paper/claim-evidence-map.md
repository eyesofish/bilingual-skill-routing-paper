# Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| The benchmark contains 26 skills and 51 queries. | `data/catalog.json`, `data/queries.json`, `results/baseline.json` | Supported |
| Both development and test sets include English, Chinese, and mixed-language prompts. | `data/queries.json` | Supported |
| Bilingual augmentation improves weighted-token Hit@1 from 53.6% to 75.0%. | `results/baseline.json` | Supported for this benchmark |
| Bilingual augmentation improves weighted-token routing accuracy from 52.9% to 73.5%. | `results/baseline.json` | Supported for this benchmark |
| Bilingual RRF reaches 85.7% Hit@1. | `results/baseline.json` | Supported for this benchmark |
| The current RRF configuration cannot safely abstain. | Test abstention accuracy is 0% in both RRF runs. | Supported for this split |
| Every evaluated method has sub-millisecond median latency on this machine. | `results/baseline.json` records 0.16-0.62 ms median latency. | Supported on the recorded machine |
| Bilingual metadata matters more than retrieval complexity in small registries. | Current weighted-token augmentation gain is larger than replacing original weighted-token with original RRF. | Preliminary; needs larger corpus and confidence intervals |
| The method generalizes to other users and registries. | No external-user or large public-corpus evaluation yet. | Needs evidence |
| Better skill retrieval improves end-to-end agent task success. | No downstream task execution experiment yet. | Needs evidence |
| This is the first bilingual skill-routing benchmark. | Quick literature search found related Chinese tool-use work but no direct match. | Do not claim without a systematic review |

## Highest-risk claims

1. **Novelty:** keep the wording narrow until a systematic literature review is
   complete.
2. **Generalization:** current results apply only to this 26-skill,
   author-written benchmark.
3. **Safety:** good Hit@1 does not justify automatic activation when abstention
   is weak.
4. **Latency:** the reported values exclude model loading because all current
   methods are deterministic sparse methods.

