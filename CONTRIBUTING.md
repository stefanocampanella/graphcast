# Contributing to Moby-Dick

Contributions are welcome as GitHub pull requests into the `develop` branch, one branch and pull request per change.
Every pull request is reviewed before it is merged.

- **Tests:** new code comes with pytest tests in `tests/` (`test_<module>.py`, shared fixtures in
  `tests/conftest.py`). They run on CPU with synthetic data, with no credentials or network. Each test has a docstring
  saying what it checks and why.
- **Checks:** `pre-commit run --all-files` runs ruff (2-space indentation, 120 columns), ShellCheck, `reuse lint` and
  ty. GitHub Actions runs them, and the tests with at least 80 % coverage, on every pull request.
- **Licensing:** contributions are licensed under the Apache License, Version 2.0. New files start with SPDX
  copyright and license headers, as the existing files do (`REUSE.toml` covers files that take no comments). Code copied from GraphCast keeps DeepMind's copyright line, says that it was modified, and is listed in
  `NOTICE`.
- **Portability:** `scripts/` and `src/` hold no cluster-specific paths or settings; those go under
  `clusters/<name>/`, and user-specific settings (e.g. credentials) come from environment variables.
