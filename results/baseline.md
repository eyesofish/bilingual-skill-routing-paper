# Baseline Results

- Catalog: 26 skills
- Queries: 51 total (17 dev, 34 test)
- Latency repeats per query: 100
- Generated: 2026-08-27T06:40:18.224852+00:00

| Configuration | Hit@1 | Hit@3 | MRR | Abstention | Routing accuracy | Median ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| original/weighted-token | 53.6% | 71.4% | 0.669 | 83.3% | 52.9% | 0.4052 | 0.4352 |
| original/char-ngram | 64.3% | 82.1% | 0.736 | 0.0% | 52.9% | 0.1603 | 0.3549 |
| original/rrf-fusion | 60.7% | 82.1% | 0.723 | 0.0% | 50.0% | 0.5805 | 0.7752 |
| bilingual-augmented/weighted-token | 75.0% | 96.4% | 0.856 | 66.7% | 73.5% | 0.4417 | 0.4703 |
| bilingual-augmented/char-ngram | 75.0% | 92.9% | 0.845 | 33.3% | 58.8% | 0.1631 | 0.3508 |
| bilingual-augmented/rrf-fusion | 85.7% | 92.9% | 0.900 | 0.0% | 70.6% | 0.6194 | 0.8206 |

Thresholds are selected only on the development split. Hit@k and MRR use
positive test queries. Abstention accuracy uses negative test queries.
Routing accuracy combines positive selection and negative abstention.
