# Multilingual Query Translation Audit

- Task count: **73**
- Corrected rows: **73**
- Validation target: `data/skillrouter_queries_multilingual.jsonl`

## Correction categories

- Restored omitted steps, conditions, negations, and numeric constraints.
- Corrected semantic inversions and inaccurate technical translations.
- Removed invented examples, outputs, and requirements.
- Restored exact filenames, paths, code symbols, inline literals, fenced code blocks, and output schemas.
- Kept `query_mixed` complete and substantially bilingual with natural Chinese instructions and unchanged English technical identifiers.
- Preserved every `query_en` value and all source metadata.

## Corrected task IDs

`3d-scan-calc`, `adaptive-cruise-control`, `azure-bgp-oscillation-route-leak`,
`citation-check`, `civ6-adjacency-optimizer`,
`crystallographic-wyckoff-position-analysis`, `dapt-intrusion-detection`,
`data-to-d3`, `dialogue-parser`, `dynamic-object-aware-egomotion`,
`earthquake-phase-association`, `earthquake-plate-calculation`,
`econ-detrending-correlation`, `energy-ac-optimal-power-flow`,
`energy-market-pricing`, `enterprise-information-search`,
`exoplanet-detection-period`, `find-topk-similiar-chemicals`,
`fix-build-agentops`, `fix-build-google-auto`, `fix-druid-loophole-cve`,
`fix-erlang-ssh-cve`, `flink-query`, `flood-risk-analysis`,
`gh-repo-analytics`, `glm-lake-mendota`, `gravitational-wave-detection`,
`grid-dispatch-operator`, `hvac-control`, `invoice-fraud-detection`,
`jax-computing-basics`, `jpg-ocr-stat`, `lab-unit-harmonization`,
`lake-warming-attribution`, `latex-formula-extraction`, `lean4-proof`,
`manufacturing-codebook-normalization`,
`manufacturing-equipment-maintenance`, `manufacturing-fjsp-optimization`,
`mario-coin-counting`, `mars-clouds-clustering`,
`multilingual-video-dubbing`, `organize-messy-files`, `paper-anonymizer`,
`parallel-tfidf-search`, `pddl-tpp-planning`,
`pedestrian-traffic-counting`, `pg-essay-to-audiobook`,
`powerlifting-coef-calc`, `python-scala-translation`,
`quantum-numerical-simulation`, `r2r-mpc-control`,
`react-performance-debugging`, `scheduling-email-assistant`,
`sec-financial-report`, `seismic-phase-picking`, `setup-fuzzing-py`,
`simpo-code-reproduction`, `software-dependency-audit`,
`speaker-diarization-subtitles`, `spring-boot-jakarta-migration`,
`suricata-custom-exfil`, `syzkaller-ppdev-syzlang`,
`taxonomy-tree-merge`, `threejs-structure-parser`, `threejs-to-obj`,
`travel-planning`, `trend-anomaly-causal-inference`,
`video-filler-word-remover`, `video-silence-remover`,
`video-tutorial-indexer`, `virtualhome-agent-planning`,
`xlsx-recover-data`.

## Remaining limitations

- Product names, API names, output labels, and other exact-match technical terms
  intentionally remain in English where translating them could break execution.
- Validation checks structural integrity and source parity; it does not execute the
  underlying 73 engineering tasks or verify their domain-specific outputs.
- No known translation omission, semantic change, invented requirement, broken
  identifier, or invalid JSON remains after this audit.
