# AeroPick: Oracle-Free Selection of LLM-Generated Aerospace Code

AeroPick selects among several LLM-generated implementations of an embedded-C function **without a test oracle**.
The model under test writes input-only test scenarios; a generic harness runs every candidate on them; candidates are
grouped by observed behavior. Where groups diverge, a reasoning-mode LLM decides from the requirement which behavior is
correct (execution judge); where no divergence appears, the reasoning-mode LLM compares the candidates' code against the
requirement (code judge).

This repository contains the method code, the **gnc100** benchmark (100 attitude determination and control tasks built
from textbooks and standards), and the generated candidate pools and cached LLM outputs for gnc100 with two models
(DeepSeek-V4-Flash, Qwen3.6-35B-A3B). The two industrial datasets used in the paper are not released.

## Layout
| Path | Content |
|---|---|
| `aeropick/llm_tests.py` | input-only scenario generation, generic harness, behavioral grouping |
| `aeropick/llm_judge.py` | execution judge (reasoning mode, project-convention glossary) |
| `aeropick/code_judge.py` | code judge |
| `aeropick/boundary_probe.py` | candidate loading, random probes, grouping utilities |
| `utils/`, `eval/` | LLM client, probe harness builder, output comparison, evaluator (Docker + cppcheck MISRA addon) |
| `utils/conventions_zh.txt` | the ten project conventions given to the execution judge |
| `dataset/gnc-adcs-100.jsonl` | gnc100: requirement, C context, reference solution, hidden test per task |
| `logs/cache/<model>/gnc100/pool{1,2,3}/` | three independent candidate pools (3 candidates per task), `_eval.json` = hidden-test and rule-check results |
| `logs/cache/<model>/gnc100/_llmtests/`, `_bprobe/` | generated scenarios, harnesses, grouping results, selection results |
| `logs/cache/<model>/gnc100/_judge_think_gloss/`, `_codejudge/` | cached judge outputs |
| `scripts/reproduce_gnc100.py` | recomputes the gnc100 numbers of the paper from the caches |

## Reproduce the gnc100 numbers (no API key, no Docker)
```
python3 scripts/reproduce_gnc100.py
```
Expected: AeroPick 69.4 ± 1.8 (DeepSeek-V4-Flash) and 40.5 ± 2.8 (Qwen3.6-35B-A3B) safe-pass@1 over three runs.

## Re-run the selection
Requires Python 3.9+, `requests`, Docker, and the evaluator image (`docker build -t aerorhf-eval:latest eval/docker`).
API keys are read from `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY`. Cached LLM outputs are reused when present.
```
MODEL_DIR=deepseek-v4-flash LLM_MODEL=deepseek-v4-flash python3 aeropick/llm_tests.py gnc100 gnc-adcs-100.jsonl
MODEL_DIR=deepseek-v4-flash LLM_MODEL=deepseek-v4-flash JUDGE_THINK=1 JUDGE_GLOSS=1 python3 aeropick/llm_judge.py gnc100 gnc-adcs-100.jsonl
MODEL_DIR=deepseek-v4-flash LLM_MODEL=deepseek-v4-flash python3 aeropick/code_judge.py gnc100 gnc-adcs-100.jsonl
```

## Key execution rules
- **Output comparison:** two values match if `|a-b| <= 1e-4·max(|a|,|b|) + 1e-9`; clearly nonzero values with opposite signs
  differ; infinities match only the same infinity; NaN matches only NaN. Candidates form one group only if they print the
  same set of fields and all fields match. Each scenario runs in a forked process with a 2 s alarm; a crash or timeout is
  recorded as an output.
- **Same inputs for all candidates:** one harness per pool; scenario statements that fail to compile are removed from the
  shared harness, so every candidate runs the identical scenarios.
- **Scoring a chosen group:** the expectation of a uniform pick inside the group.
