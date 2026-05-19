# Claude's Development Workflow

Rules Claude follows when working on this project.

## Branch & PR rules

- **Never commit or push directly to `main`**
- Every fix, feature, or improvement gets its own branch
- Branch naming: `improve/<topic>`, `fix/<topic>`, `feat/<topic>`
- Always open a PR to `main` and notify the user — never self-merge
- PR description must include a summary of changes and a test plan

## Before opening a PR

- `poetry run pytest tests/ -v` — all tests must pass
- `poetry run ruff check src/ tests/` — no lint errors
- New behaviour must have new or updated tests

## Proactive improvements

When Claude identifies a quality gap (missing test coverage, a bug risk, a design smell), it will:

1. Describe the issue and proposed fix to the user
2. Wait for approval
3. Create a branch, implement, open a PR
4. Let the user know the PR is ready for review

## What NOT to do

- Do not push to `main` under any circumstances
- Do not merge PRs without user approval
- Do not add unnecessary abstractions, comments, or error handling
- Do not implement features the user has not approved
