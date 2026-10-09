# Contributing

Keep each pull request focused on one correction or improvement. Describe the
problem, resulting behavior and validation. Project documentation, interface text
and commit messages are in English.

## Branches and merging

`main` is the only permanent branch. Use a short-lived descriptive branch:

| Prefix | Purpose |
|---|---|
| `fix/` | Correct a defect |
| `feature/` | Add useful behavior |
| `study/` | Change or extend the research protocol |
| `docs/` | Improve documentation |
| `maintenance/` | Maintain dependencies or CI |

Start in the existing checkout with a clean working tree:

```bash
git switch main
git pull --ff-only
git switch -c fix/short-description

# Make and check one focused change.
python -m pytest -q
git diff --check
git add <changed-files>
git diff --cached
git commit -m "Describe the completed change"
git push -u origin HEAD
gh pr create --base main
```

Wait for all checks on the current PR revision, then merge:

```bash
gh pr checks <number> --watch
gh pr merge <number> --squash --delete-branch
git switch main
git pull --ff-only
git fetch --prune
```

Replace angle-bracket placeholders with actual paths or the PR number.
The repository requires an up-to-date PR and all five CI checks: both grouping
modes on Ubuntu/macOS, plus the interactive demo. No additional reviewer is
required for this single-maintainer workflow. Unresolved review conversations
must be resolved before merging.

Squash merging records one logical change in `main`; merged remote branches are
deleted automatically. Main-branch force pushes and deletion are disabled.
After merging, remove any remaining local branch only after verifying that its
changes are included. Do not delete branches containing unfinished work.

## Validation

Use the pinned Python 3.12 environment. Install `requirements-app.txt` when editing
the demo, and run `tests/test_demo.py` and `tests/test_app.py` as well as relevant
core tests. Warnings are treated as errors.

For changes to training, grouping, attacks, thresholds or metric calculation,
run a real-data quick experiment in a new ignored directory and independently
verify every attack row. Check both grouping modes when the change affects them.
See the [validation report](docs/VALIDATION.md) and
[experiment protocol](docs/PROTOCOL.md).

New local runs belong under `results/local/`. Keep trained binaries, caches,
virtual environments and private messages out of Git. Published results in
`results/full/` and `results/template/` carry source and data hashes; changing
their training code requires a documented new run and consistent artifacts.
Never silently edit reported metrics or tune the frozen model using test results.

## Documentation and dependencies

Keep the README concise; put detailed methodology in `docs/PROTOCOL.md`.
Update the model card and manuscript when reported claims change. Preserve
original historical evidence and identify exploratory follow-up analyses.

Runtime and optional notebook/app dependencies have separate locks. Resolve
updates deliberately, run `python -m pip check`, and validate the affected
workflows. The app lock includes the pinned runtime requirements:

```bash
uv pip compile requirements-app.in --no-header --no-annotate --output-file requirements-app.txt
```
