# SkillRouter Multilingual Results

- Candidate skills: 79141
- Tasks: 73
- Query variants: 219
- Tier: hard

| Method | EN H@1 | ZH H@1 | ZH+Inst H@1 | EN R@10 | ZH R@10 | ZH+Inst R@10 | ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| word-tfidf | 37.0% | 8.2% | 8.2% | 46.8% | 17.7% | 17.7% | 5.7 |
| word-bm25 | 46.6% | 19.2% | 12.3% | 46.0% | 21.7% | 17.1% | 5.6 |
| char-tfidf | 49.3% | 21.9% | 24.7% | 50.1% | 31.4% | 29.4% | 14.8 |
| multilingual-e5-small | 13.7% | 0.0% | 0.0% | 15.5% | 0.0% | 0.0% | 12.0 |
| skillrouter-emb-0.6b | 45.2% | 41.1% | 26.0% | 50.5% | 43.8% | 38.9% | 140.8 |
| skillrouter+char-rrf | 49.3% | 42.5% | 39.7% | 54.9% | 46.0% | 42.1% | 155.6 |

Hit@1 confidence intervals use 10,000 task-level bootstrap samples with seed 20270827. Source-excluded and generic-only tasks are omitted.
