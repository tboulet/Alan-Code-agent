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
   python -m build
   python -m twine check dist/*
   ```

## Publish

Publishing and tagging affect external state. Do them only with explicit release-owner approval:

```bash
python -m twine upload dist/*
git tag v1.3.0
git push origin v1.3.0
```

After publishing, install the wheel in a fresh virtual environment and verify `alancode --version` plus one `ScriptedBackend` smoke turn.
