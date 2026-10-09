---
name: code-review
description: Review a change or pull request in geonnax against CODE_REVIEW.md, the contracts in AGENTS.md (Equinox modules and init, the basis contract, numerics and initialisation, the public API), the boundaries with pyrox and kernellib, and reuse of existing building blocks.
---

# Code review

Read "Boundaries", "Reuse before you write" and "The contracts" in
[`AGENTS.md`](../../../AGENTS.md); for every function, class or module the
diff adds, search [`docs/api/capabilities.md`](../../../docs/api/capabilities.md)
for an existing equivalent; then apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md)
and report in its format.
