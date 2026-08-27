# SkillRouter Error Analysis

## word-tfidf
- en: 27/73 Hit@1; 71 unique top predictions; largest hub `distractor/dist_pddl-skills_de660494` appears 2 times; length correlation rho=0.125 (p=0.294).
- zh: 6/73 Hit@1; 57 unique top predictions; largest hub `development/siyuan-notes-2234839-siyuan-notes-skill` appears 8 times; length correlation rho=-0.192 (p=0.104).
- zh_instruction: 6/73 Hit@1; 57 unique top predictions; largest hub `development/siyuan-notes-2234839-siyuan-notes-skill` appears 8 times; length correlation rho=-0.192 (p=0.104).

## word-bm25
- en: 34/73 Hit@1; 67 unique top predictions; largest hub `gt/simulation-metrics` appears 3 times; length correlation rho=0.170 (p=0.15).
- zh: 14/73 Hit@1; 69 unique top predictions; largest hub `other/dotfiles-commit-erwinkn-ai-config` appears 2 times; length correlation rho=-0.183 (p=0.121).
- zh_instruction: 9/73 Hit@1; 57 unique top predictions; largest hub `other/csv-workbench` appears 6 times; length correlation rho=-0.156 (p=0.187).

## char-tfidf
- en: 36/73 Hit@1; 68 unique top predictions; largest hub `gt/seismic-picker-selection` appears 2 times; length correlation rho=0.084 (p=0.481).
- zh: 16/73 Hit@1; 65 unique top predictions; largest hub `development/unit-executor` appears 4 times; length correlation rho=-0.016 (p=0.895).
- zh_instruction: 18/73 Hit@1; 65 unique top predictions; largest hub `design/technical-constraints-dylantarre-animation-principles` appears 4 times; length correlation rho=0.044 (p=0.713).

## multilingual-e5-small
- en: 10/73 Hit@1; 48 unique top predictions; largest hub `data/file-analysis` appears 10 times; length correlation rho=0.042 (p=0.727).
- zh: 0/73 Hit@1; 48 unique top predictions; largest hub `development/posts-gongfukangee-gongfukangee-gith` appears 12 times; length correlation rho=0.000 (p=1).
- zh_instruction: 0/73 Hit@1; 47 unique top predictions; largest hub `development/posts-gongfukangee-gongfukangee-gith` appears 7 times; length correlation rho=0.000 (p=1).

## skillrouter-emb-0.6b
- en: 33/73 Hit@1; 71 unique top predictions; largest hub `other/debug-zhiruifeng-localagentcrew` appears 2 times; length correlation rho=-0.142 (p=0.232).
- zh: 30/73 Hit@1; 71 unique top predictions; largest hub `gt/economic-dispatch` appears 2 times; length correlation rho=-0.182 (p=0.123).
- zh_instruction: 19/73 Hit@1; 69 unique top predictions; largest hub `development/codex-reasoning` appears 2 times; length correlation rho=-0.106 (p=0.372).

## skillrouter+char-rrf
- en: 36/73 Hit@1; 70 unique top predictions; largest hub `gt/fjsp-baseline-repair-with-downtime-and-policy` appears 2 times; length correlation rho=-0.031 (p=0.797).
- zh: 31/73 Hit@1; 71 unique top predictions; largest hub `gt/pddl-skills` appears 2 times; length correlation rho=0.000 (p=1).
- zh_instruction: 29/73 Hit@1; 70 unique top predictions; largest hub `development/codex-reasoning` appears 2 times; length correlation rho=-0.023 (p=0.845).

## Pairwise task changes
- zh:char-tfidf->skillrouter-emb-0.6b: 21 gains, 7 losses, 9 shared hits.
  - gains: 3d-scan-calc, dapt-intrusion-detection, dialogue-parser, earthquake-phase-association, exoplanet-detection-period, flood-risk-analysis, glm-lake-mendota, gravitational-wave-detection, lab-unit-harmonization, latex-formula-extraction, manufacturing-codebook-normalization, manufacturing-equipment-maintenance, paper-anonymizer, pedestrian-traffic-counting, react-performance-debugging, sec-financial-report, seismic-phase-picking, suricata-custom-exfil, threejs-structure-parser, video-filler-word-remover, xlsx-recover-data
  - losses: pddl-tpp-planning, powerlifting-coef-calc, software-dependency-audit, spring-boot-jakarta-migration, threejs-to-obj, trend-anomaly-causal-inference, virtualhome-agent-planning
- zh:skillrouter-emb-0.6b->skillrouter+char-rrf: 9 gains, 8 losses, 22 shared hits.
  - gains: dynamic-object-aware-egomotion, manufacturing-fjsp-optimization, pddl-tpp-planning, powerlifting-coef-calc, python-scala-translation, threejs-to-obj, trend-anomaly-causal-inference, video-silence-remover, virtualhome-agent-planning
  - losses: dapt-intrusion-detection, exoplanet-detection-period, gravitational-wave-detection, paper-anonymizer, pedestrian-traffic-counting, react-performance-debugging, seismic-phase-picking, xlsx-recover-data
- zh_instruction:char-tfidf->skillrouter-emb-0.6b: 14 gains, 13 losses, 5 shared hits.
  - gains: 3d-scan-calc, dialogue-parser, earthquake-phase-association, econ-detrending-correlation, glm-lake-mendota, gravitational-wave-detection, lab-unit-harmonization, manufacturing-codebook-normalization, manufacturing-equipment-maintenance, paper-anonymizer, react-performance-debugging, seismic-phase-picking, suricata-custom-exfil, xlsx-recover-data
  - losses: earthquake-plate-calculation, energy-ac-optimal-power-flow, grid-dispatch-operator, latex-formula-extraction, pddl-tpp-planning, powerlifting-coef-calc, setup-fuzzing-py, software-dependency-audit, spring-boot-jakarta-migration, taxonomy-tree-merge, threejs-to-obj, trend-anomaly-causal-inference, virtualhome-agent-planning
- zh_instruction:skillrouter-emb-0.6b->skillrouter+char-rrf: 16 gains, 6 losses, 13 shared hits.
  - gains: data-to-d3, dynamic-object-aware-egomotion, earthquake-plate-calculation, energy-ac-optimal-power-flow, grid-dispatch-operator, latex-formula-extraction, pddl-tpp-planning, powerlifting-coef-calc, python-scala-translation, setup-fuzzing-py, software-dependency-audit, taxonomy-tree-merge, threejs-structure-parser, threejs-to-obj, video-filler-word-remover, virtualhome-agent-planning
  - losses: 3d-scan-calc, econ-detrending-correlation, gravitational-wave-detection, paper-anonymizer, seismic-phase-picking, xlsx-recover-data
