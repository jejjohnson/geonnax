# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/api/capabilities.md` (every public geonnax
  name, plus the shared private helpers) before writing a layer, a basis, an
  encoder, an init rule or a helper; the "Reuse before you write" table in
  `AGENTS.md` maps the usual hand-rolled code to what already exists.
- **Downstream first.** pyrox and kernellib import these names and swap
  these parameter fields; before renaming anything, check how they use it
  (search jejjohnson/pyrox and jejjohnson/kernellib for the name) and follow
  "Deprecate, don't break" in `AGENTS.md`.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building: `add-layer`, `add-model`, `add-basis`, `add-encoder`,
    `add-uncertainty-core`, `add-conditioner`, `add-notebook`;
  - shipping: `pre-pr-check`, `geonnax-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `geonnax-review`; run them on any diff that adds code, before committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/api/capabilities.md` (or something that belongs in pyrox or
    kernellib)?
  - `numerics-reviewer`: static arrays, keys, shape conventions, dtypes,
    traced control flow, NaN gradients at singular points, init scales,
    power iteration and Cholesky stability, under-resolved grids.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
