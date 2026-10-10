---
applyTo: "**"
---

# Code Review Instructions

When performing code review, use `/CODE_REVIEW.md` as the source of truth for:

- The review checklist: reuse and boundaries, Equinox modules and `init`,
  the basis contract, numerics and initialisation, the public API and
  downstream compatibility, docs, tests, idioms and dependencies
- geonnax-specific ❌/✅ examples
- Output format and priority levels
- Suggestion type emojis and review tone

Read "Boundaries", "Reuse before you write" and "The contracts" in
`/AGENTS.md` first.

Key principles:
- Most defects worth finding are about modules and numerics (a static array,
  a reused key, a batch axis baked in, an init scale off by ω, a NaN gradient
  at a coincident point) or boundaries (numpyro, a re-implemented layer, a
  renamed field pyrox swaps), not style.
- Don't worry about formatting — CI (ruff format, pre-commit) handles that automatically.
- Be **constructive** and **specific**. Acknowledge good patterns with 👍.
- Every suggestion must include a concrete alternative, not just criticism.
