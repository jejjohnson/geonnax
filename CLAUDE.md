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
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
