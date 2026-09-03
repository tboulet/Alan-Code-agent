"""Provenance must describe the running source, not the install metadata."""

import alancode
from alancode.__version__ import __version__
from alancode import provenance


def test_reports_the_imported_source_not_package_metadata():
    # The bug this exists for: an editable install keeps the dist-info version
    # from install time, so metadata can disagree with the code by many
    # releases. Provenance must follow the source.
    info = alancode.get_provenance()
    assert info["version"] == __version__
    assert info["path"].endswith("alancode")


def test_git_fields_are_populated_in_a_checkout():
    info = alancode.get_provenance()
    assert info["git_sha"] and len(info["git_sha"]) >= 7
    assert isinstance(info["git_dirty"], bool)


def test_git_fields_are_none_outside_a_checkout(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda repo, *args: None)
    info = provenance.get_provenance()
    assert info["version"] == __version__
    assert info["git_sha"] is None
    assert info["git_dirty"] is None


def test_a_missing_git_binary_never_raises(monkeypatch):
    # Patch subprocess itself, not _git: the error handling lives inside _git,
    # so patching _git would test nothing.
    def explode(*args, **kwargs):
        raise FileNotFoundError("no git on PATH")

    monkeypatch.setattr(provenance.subprocess, "run", explode)
    info = provenance.get_provenance()
    assert info["version"] == __version__
    assert info["git_sha"] is None


def test_a_git_timeout_never_raises(monkeypatch):
    import subprocess as sp

    def timeout(*args, **kwargs):
        raise sp.TimeoutExpired(cmd="git", timeout=provenance.GIT_TIMEOUT_S)

    monkeypatch.setattr(provenance.subprocess, "run", timeout)
    assert provenance.get_provenance()["git_sha"] is None


def test_string_form_marks_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(
        provenance, "get_provenance",
        lambda: {"version": "9.9.9", "git_sha": "abc123456789", "git_dirty": True},
    )
    assert provenance.provenance_string() == "9.9.9 (abc123456789, dirty)"


def test_string_form_omits_git_when_absent(monkeypatch):
    monkeypatch.setattr(
        provenance, "get_provenance",
        lambda: {"version": "9.9.9", "git_sha": None, "git_dirty": None},
    )
    assert provenance.provenance_string() == "9.9.9"
