# Abstract Draft

Agent skills are reusable instruction packages that an agent selects from a
registry before loading the full procedure. Existing work has studied routing
at very large scale, but personal registries create a different problem:
dozens of overlapping skills, short informal requests, and frequent mixing of
English technical terms with Chinese intent. We present an initial,
reproducible benchmark for this setting. The benchmark contains 26 skills and
51 human-written requests split into English, Chinese, mixed-language, and
no-skill cases. We compare original metadata with a bilingual alias treatment
using three CPU-only methods: weighted token overlap, character n-gram TF-IDF,
and reciprocal-rank fusion. Thresholds for abstention are selected only on a
development split. In the first test run, bilingual metadata raises weighted
token Hit@1 from 53.6% to 75.0% and combined routing accuracy from 52.9% to
73.5%, while keeping median query latency below 0.45 ms. Rank fusion reaches
85.7% Hit@1 with bilingual metadata, but fails to abstain on no-skill requests.
These results suggest that metadata language coverage can matter more than a
more complex sparse ranker in a small registry, and that retrieval quality and
safe abstention must be evaluated separately. The current benchmark is small
and author-written, so the results are a starting point rather than a general
claim about agent skill routing.

