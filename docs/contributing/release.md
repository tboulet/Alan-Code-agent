# Release process

How to cut a new version of `alancode` and publish it to PyPI.

## Versioning

Alan Code uses [SemVer](https://semver.org/):

- **MAJOR** — breaking changes to the public API (`AlanCodeAgent` class, CLI flags, settings.json keys with data-loss semantics, tool schemas).
- **MINOR** — new features, backwards-compatible additions.
- **PATCH** — bug fixes.

The version is stored in `alancode/__version__.py`:

```python
__version__ = "1.0.0"
```

Hatchling reads this at build time (`pyproject.toml::tool.hatch.version`).

## Pre-release checklist

Before cutting a release:

1. **All tests pass** locally: `pytest -x -q`.
2. **Lint is clean**: `ruff check .`.
3. **Manual smoke test** against a real model:
   ```bash
   alancode --provider litellm --model openrouter/google/gemini-2.5-flash
   > /status
   > write a short test
   > /exit
   ```
4. **GUI smoke test**: `alancode --gui`, open browser, run a turn, check Git Tree + LLM Perspective panels.
5. **Examples still work**:
   ```bash
   python examples/example_1_cli_agent.py
   python examples/example_2_auto_fix_loop/run_alan.py --scripted
   ```
6. **Bump the version** in `alancode/__version__.py`.
7. **Update changelog** (see below).

## Changelog

Keep a `CHANGELOG.md` at the repo root. Format:

```markdown
# Changelog

## [0.2.0] — 2026-05-01

### Added
- `/provider` command for switching providers mid-session.
- Support for OpenRouter's new cache API.

### Changed
- `max_turns` renamed to `max_iterations_per_turn`.
- Memory default is now `off`.

### Fixed
- Hook timeout no longer silently allows (falls back to `ask`).
- GLM tool-call regex now requires the closing tag.

### Removed
- `/undo` command. Use `/revert` instead.
```

One entry per release. Current unreleased work goes under an `[Unreleased]` header that gets renamed on release.

## Building

Alan uses [hatchling](https://hatch.pypa.io/) as the build backend (declared in `pyproject.toml`).

```bash
pip install build
python -m build
```

Produces:
- `dist/alancode-<version>.tar.gz` (source distribution)
- `dist/alancode-<version>-py3-none-any.whl` (wheel)

Inspect the wheel contents:

```bash
unzip -l dist/alancode-<version>-py3-none-any.whl
```

Should include `alancode/` with all `.py` files and `alancode/gui/static/` with HTML/JS/CSS. If static files are missing, check that `pyproject.toml` packages them (hatchling includes package data by default; verify with the `unzip -l`).

## Publishing to PyPI

First time only — configure credentials:

```bash
pip install twine
# Set up ~/.pypirc or use API token env var TWINE_PASSWORD
```

Test first on TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Verify:

```bash
pip install --index-url https://test.pypi.org/simple/ alancode
alancode --version
```

Then publish to real PyPI:

```bash
python -m twine upload dist/*
```

## Git tag

After publish:

```bash
git tag v0.2.0
git push --tags
```

Tag format: `vMAJOR.MINOR.PATCH`.

## GitHub release

Create a GitHub release referencing the tag. Copy the changelog entry into the release notes.

## Post-release

1. Bump `__version__` to the next dev version (e.g. `0.2.1-dev`) if you use that pattern, or leave at the released version until the next bump.
2. Open an `[Unreleased]` section in `CHANGELOG.md` for the next cycle.
3. Announce the release (if applicable — blog post, X, Discord, etc.).

## Hotfix process

For urgent bug fixes:

1. Branch from the latest release tag: `git checkout -b hotfix/0.2.1 v0.2.0`.
2. Fix the bug, add a regression test.
3. Bump patch version (`0.2.0` → `0.2.1`).
4. Run the full checklist.
5. Publish.
6. Merge the hotfix branch into `main`.

## Dependency updates

`pyproject.toml` declares minimum versions. Periodically:

1. Review what's pinned.
2. Test with newer versions of `anthropic`, `litellm`, `rich`, `prompt_toolkit`, `fastapi`, `uvicorn`.
3. Bump minimums if a new feature we rely on requires it.
4. Avoid upper pins unless a specific version is known broken.

## Related

- [contributing/development.md](development.md) — dev setup.
- [contributing/testing.md](testing.md) — test organisation.
