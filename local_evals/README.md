# Nebula Antigravity Test Suite

This folder contains a lightweight 60-query test suite for local LLM evaluation
against the 35B Worker model.

The suite is intentionally self-contained rather than Docker/SWE-bench-harness
based. Coding queries give enough miniature repository context for the model to
produce a concrete patch or implementation plan without fetching upstream
projects. Stability queries are inspired by the local Perplexity Python client
repo and test hallucination resistance, API boundary discipline, and error
handling.

The split is fixed:

- 10 easy
- 10 moderate
- 10 difficult
- 10 very_difficult
- 20 stability

Run a validation-only check:

```bash
python3 local_evals/run_nebula_antigravity_eval.py --validate-only
```

Run a smoke query against the current Worker:

```bash
python3 local_evals/run_nebula_antigravity_eval.py --limit 1
```

Run all 60:

```bash
python3 local_evals/run_nebula_antigravity_eval.py
```

Outputs are JSONL records under `local_evals/results/` by default. The runner
uses the canonical local inference router at `http://127.0.0.1:11888/v1`.
The runner resolves the canonical Worker model from
`/mnt/f/Production/antigravity/brain/model_routing_policy.json` unless
`--model` is provided. Each run prints aggregate TPS, token, timing,
competency, truncation, continuation, and error-rate measurements at the end.
