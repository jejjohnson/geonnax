---
name: squash-commit
description: Write a Conventional-Commits squash commit message for a geonnax pull request from its commits and description. Use when asked for a squash / merge commit message.
---

Generate a squash commit message for a GitHub PR.

## Instructions

1. If an argument is provided (PR number or URL), fetch the PR details
   (`gh pr view`, or the GitHub MCP tools). Otherwise, detect the current
   branch and find its open PR.
2. Fetch all individual commit messages in the PR (`gh api` or `git log`).
3. Combine them into a single squash commit message following these rules.

### Format

```
<type>(<scope>): <concise summary of the overall change>

<1-3 sentence description combining the key changes from all commits.
Focus on the "why" and overall effect, not per-commit details.>

Co-authored-by: <preserve all unique Co-authored-by lines from the individual commits>
```

### Rules

- Conventional Commits types: `feat`, `fix`, `docs`, `style`, `refactor`,
  `perf`, `test`, `build`, `ci`, `chore`, `revert`; a lowercase subject (CI
  validates PR titles); the scope is usually the module or family touched
  (`layers`, `basis`, `encoders`, `fno`, `sfno`, `wno`, `mswt`, `unet`,
  `siren`, `mfn`, `conditioning`, `sngp`, `ensemble`, `heteroscedastic`,
  `spectral-norm`, `spherical-wavelets`, `api`, `docs`, `deps`, …).
  release-please (`release-please-config.json`) builds `CHANGELOG.md` from
  `feat`, `fix`, `perf` and `revert` commits only, so the type decides
  whether a change is listed; `chore(main): release X.Y.Z` is its own.
- The type/scope reflect the dominant change across all commits; mention the
  rest in the body.
- A breaking change keeps its `!` and `BREAKING CHANGE:` footer.
- Collapse redundant or incremental commits into one coherent description.
- Preserve ALL unique `Co-authored-by` lines.
- Keep the summary line under 72 characters.
- Do NOT list the individual commit messages as bullet points — this is a
  squash, not a merge.

### Output

Print ONLY the final squash commit message in a code block so the user can
copy it directly. Do not add explanation before or after.
