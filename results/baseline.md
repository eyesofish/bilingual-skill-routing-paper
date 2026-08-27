# Baseline Results

- Catalog: 26 skills
- Queries: 51 total (17 dev, 34 test)
- Latency repeats per query: 100
- Generated: 2026-08-27T10:46:20.914625+00:00

| Configuration | Hit@1 | Hit@3 | MRR | Abstention | Routing accuracy | Median ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| original/weighted-token | 53.6% | 71.4% | 0.669 | 83.3% | 52.9% | 0.3508 | 0.3804 |
| original/char-ngram | 64.3% | 82.1% | 0.736 | 0.0% | 52.9% | 0.1243 | 0.2744 |
| original/rrf-fusion | 60.7% | 82.1% | 0.723 | 0.0% | 50.0% | 0.4945 | 0.6482 |
| bilingual-augmented/weighted-token | 71.4% | 89.3% | 0.806 | 66.7% | 64.7% | 0.3830 | 0.4123 |
| bilingual-augmented/char-ngram | 64.3% | 82.1% | 0.737 | 100.0% | 55.9% | 0.1236 | 0.2748 |
| bilingual-augmented/rrf-fusion | 75.0% | 82.1% | 0.804 | 0.0% | 61.8% | 0.5290 | 0.6755 |

Thresholds are selected only on the development split. Hit@k and MRR use
positive test queries. Abstention accuracy uses negative test queries.
Routing accuracy combines positive selection and negative abstention.
