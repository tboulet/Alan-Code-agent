# Release process

Alan Code uses Hatchling and reads its version from `alancode/__version__.py`.

## Prepare

1. Update `alancode/__version__.py` and add the dated release notes to `CHANGELOG.md` and the README's "What's new" section.
2. Confirm public constructor, CLI, settings, and backend changes are reflected under `docs/reference/`.
3. Run the complete suite and lint:

   ```bash
   LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest tests/ -q
   ruff check .
   ```

4. Build and inspect the artifacts:

   ```bash
   VERSION=$(python -c 'from alancode import __version__; print(__version__)')
   OUT_DIR="dist/${VERSION}"
   python -m build --outdir "${OUT_DIR}"
   python -m twine check "${OUT_DIR}"/*
   ```

## Publish

Publishing and tagging affect external state. Do them only with explicit release-owner approval:

```bash
VERSION=$(python -c 'from alancode import __version__; print(__version__)')
OUT_DIR="dist/${VERSION}"
python -m twine upload "${OUT_DIR}"/*
git tag "v${VERSION}"
git push origin "v${VERSION}"
```

After publishing, install the wheel in a fresh virtual environment and verify `alancode --version` plus one `ScriptedBackend` smoke turn.
