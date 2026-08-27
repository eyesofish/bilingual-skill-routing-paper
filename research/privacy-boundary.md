# Publication and Privacy Boundary

## Allowed evidence

- Original code and tests from `eyesofishTUI`.
- User-authored skill metadata that does not contain work or private inbox data.
- The controlled skill fixtures under `eyesofishTUI/test/fixtures/skills`.
- Public papers, public repositories, and public documentation.
- Aggregate benchmark metrics and sanitized query text in this repository.

## Excluded evidence

- The leaked source mirror under `~/personal/claude-code`.
- Any design claim that depends on that leaked source.
- Copilot CLI internal implementation details, private issue or pull request
  details, unpublished work artifacts, and Microsoft-specific contribution
  records.
- Raw chat logs, email exports, work logs, interview-story inbox material,
  credentials, tokens, and personal application records.
- Absolute local paths in publishable result files.

## Claim policy

- Mark planned contributions as hypotheses until measurements support them.
- Do not claim that the benchmark is representative of all agent users.
- Do not claim novelty from an absence found only through a quick search.
- Separate retrieval quality from end-to-end agent task success.
- Report the exact catalog size, query count, split, method, threshold rule,
  machine context, and random seed for every result.

