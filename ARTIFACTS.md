# Reproducibility Artifacts

## Expected local resources

- SkillRouter hard tier: about 207 MB compressed.
- `multilingual-e5-small` metadata cache: about 116 MB.
- `SkillRouter-Embedding-0.6B` metadata cache: about 309 MB.
- Python environment: PyTorch, sentence-transformers, scikit-learn, NumPy,
  SciPy, and Matplotlib as pinned in `requirements-aaai-lock.txt`.

## Complete build

```bash
make setup
make data
make validate-data
make cache-e5
make cache-skillrouter
make experiments
make test
make submission
```

The embedding steps are one-time costs. Later experiment runs reuse the
memory-mapped `.npy` matrices under `data/external/cache/`.
Model repository revisions are pinned in `config/model_revisions.json`.
Evaluation rejects a cache unless its model revision, ordered skill-ID hash,
formatted-document hash, field format, row count, and matrix shape match the
current run.

## Expected outputs

- `results/skillrouter_multilingual.json`
- `results/skillrouter_rankings.json`
- `results/error_analysis.json`
- `paper/main.pdf`
- `paper/reproducibility-checklist.pdf`
- `paper/supplement.pdf`
- `build/aaai-submission.zip`
- `build/code-data-supplement.zip`

The submission audit rejects over-length content, non-Letter paper, Type 3 or
unembedded fonts, unresolved references, forbidden formatting commands,
identity strings, unanswered checklist items, or missing archives.
