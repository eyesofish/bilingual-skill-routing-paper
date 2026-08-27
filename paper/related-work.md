# Related Work Seeds

This is a seed list for a formal literature review. It is not yet a complete
bibliography.

## Agent skill format and loading

### Agent Skills specification

- URL: <https://agentskills.io/specification>
- Relevance: defines `SKILL.md`, required `name` and `description` metadata, and
  progressive disclosure from metadata to full instructions.
- Connection: motivates why routing quality depends on short metadata before
  the full body is loaded.

### GitHub Copilot CLI agent skills

- URL:
  <https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills>
- Relevance: documents personal and project skill locations and states that
  Copilot chooses skills from the user's prompt and the skill description.
- Connection: shows that metadata-level triggering is part of a deployed agent
  workflow.

## Skill routing

### SkillRouter: Skill Routing for LLM Agents at Scale

- Paper: <https://arxiv.org/abs/2603.22455>
- Code: <https://github.com/zhengyanzhao1997/SkillRouter>
- Relevance: evaluates about 80K skills and shows that full skill text is an
  important signal in highly overlapping registries.
- Difference: our study is CPU-only, small-registry, bilingual, metadata-design
  and abstention focused. We must not imply that it competes with SkillRouter's
  scale or neural retrieval quality.

### BM25 Skills Search

- Code: <https://github.com/n-pitabread/bm25-skills-search>
- Relevance: public deterministic skill-routing prototype with BM25, optional
  local-model selection, and explicit `NONE` abstention.
- Difference: its reported snapshot has 24 prompts. Our study adds a structured
  English, Chinese, mixed-language, and metadata-ablation design.

### SkillsBench

- Code: <https://github.com/benchflow-ai/skillsbench>
- Relevance: upstream source for the gold queries and skills used by
  SkillRouter.
- Follow-up: inspect its licenses, task construction, and language distribution
  before using any data.

## Tool awareness and selection

### MetaTool

- Paper: <https://arxiv.org/abs/2310.03128>
- Relevance: separates deciding whether to use a tool from deciding which tool
  to use.
- Connection: supports evaluating no-skill abstention separately from top-k
  retrieval.

### ToolBench

- Paper: <https://arxiv.org/abs/2307.16789>
- Code: <https://github.com/OpenBMB/ToolBench>
- Relevance: broad benchmark for tool learning and tool-use execution.
- Difference: tools expose callable APIs, while skills are instruction
  packages. The downstream execution task is related but not identical.

### CToolEval

- Paper: <https://aclanthology.org/2024.findings-acl.928/>
- Code: <https://github.com/tjunlp-lab/CToolEval>
- Relevance: Chinese benchmark for real-world API interaction and agent
  decision-making.
- Difference: it evaluates Chinese tool use, not bilingual metadata retrieval
  over personal `SKILL.md` registries.

## Router evaluation

### LLMRouterBench

- Paper: <https://arxiv.org/abs/2601.07206>
- Code: <https://github.com/ynulihao/LLMRouterBench>
- Relevance: provides routing metrics, cost tradeoffs, and a unified evaluation
  framework over model-routing tasks.
- Difference: the routed objects are models rather than skills.

## Review questions

1. Which prior benchmark includes code-switched queries rather than separate
   monolingual subsets?
2. Which work reports no-tool or no-skill calibration, not only top-k recall?
3. Which skill-routing datasets publish license-audited skill metadata?
4. Which methods report cold-start CPU latency on a laptop?
5. How should near-duplicate skills be represented in gold labels?

