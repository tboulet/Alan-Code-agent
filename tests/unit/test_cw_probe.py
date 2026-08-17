"""Tests for context-window probing, caching, and the resolution chain.

Covers: the descending probe ladder (success, refinement, ceiling
distrust, fatal errors, floor rejection), the probe cache roundtrip, the
LiteLLMBackend resolution rungs (fallback + warning, cache hit), and the
fixed Ollama /api/show parsing.
"""

import json
import logging

import pytest

import alancode.backends.cw_probe as cw_probe
from alancode.backends.cw_probe import (
    ProbeResult,
    load_cached_context_window,
    probe_context_window,
    save_cached_context_window,
)
from alancode.backends.litellm_backend import LiteLLMBackend


# ---------------------------------------------------------------------------
# Probe ladder logic (with a scripted _probe_attempt)
# ---------------------------------------------------------------------------


def script_attempts(monkeypatch, behavior):
    """Replace _probe_attempt with a scripted function.

    *behavior* maps the nominal probe size to an outcome tuple; sizes not
    in the map raise (test bug guard). Records the sequence of sizes tried.
    """
    calls = []

    async def fake_attempt(model, n_tokens, **kwargs):
        calls.append(n_tokens)
        return behavior(n_tokens)

    monkeypatch.setattr(cw_probe, "_probe_attempt", fake_attempt)
    return calls


class TestProbeLadder:
    @pytest.mark.asyncio
    async def test_descends_to_first_success(self, monkeypatch):
        # Real limit ~200k: everything above 131072 rejected.
        def behavior(n):
            if n > 200_000:
                return ("too_long", 0)
            return ("ok", n)  # backend reports the accepted size

        calls = script_attempts(monkeypatch, behavior)
        result = await probe_context_window("some/model")
        assert result.method == "probed"
        # First success at 131072; refinement at 196608 also fits (< 200k).
        assert result.value == 196_608
        # Ladder descended: 1M, 512k, 256k rejected (free), 131k paid,
        # one refinement paid.
        assert calls == [1_048_576, 524_288, 262_144, 131_072, 196_608]

    @pytest.mark.asyncio
    async def test_refinement_failure_keeps_bracket_low(self, monkeypatch):
        # Real limit 140k: refinement at 196k fails, keep 131k measurement.
        def behavior(n):
            if n > 140_000:
                return ("too_long", 0)
            return ("ok", n)

        script_attempts(monkeypatch, behavior)
        result = await probe_context_window("some/model")
        assert result.method == "probed"
        assert result.value == 131_072

    @pytest.mark.asyncio
    async def test_uses_reported_usage_not_nominal(self, monkeypatch):
        # The backend's own token count (71% of our nominal padding) is
        # the measurement, not the nominal probe size. First success at
        # 65_536, refinement at 98_304 -> int(98_304 * 0.71) = 69_795.
        def behavior(n):
            if n > 100_000:
                return ("too_long", 0)
            return ("ok", int(n * 0.71))

        script_attempts(monkeypatch, behavior)
        result = await probe_context_window("some/model")
        assert result.method == "probed"
        assert result.value == 69_795

    @pytest.mark.asyncio
    async def test_ceiling_acceptance_is_distrusted(self, monkeypatch):
        # Silent truncator (Ollama-style): accepts anything.
        def behavior(n):
            return ("ok", n)

        calls = script_attempts(monkeypatch, behavior)
        result = await probe_context_window("some/model")
        assert result.method == "ceiling_distrust"
        assert result.value is None
        assert calls == [1_048_576]  # stopped immediately

    @pytest.mark.asyncio
    async def test_fatal_error_aborts(self, monkeypatch):
        def behavior(n):
            return ("fatal", 0)

        result = await probe_context_window("some/model")
        script_attempts(monkeypatch, behavior)
        result = await probe_context_window("some/model")
        assert result.method == "failed"
        assert result.value is None

    @pytest.mark.asyncio
    async def test_everything_rejected(self, monkeypatch):
        def behavior(n):
            return ("too_long", 0)

        script_attempts(monkeypatch, behavior)
        result = await probe_context_window("some/model")
        assert result.method == "failed"
        assert result.value is None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestProbeCache:
    @pytest.fixture(autouse=True)
    def tmp_cache(self, tmp_path, monkeypatch):
        cache = tmp_path / "context_windows.json"
        monkeypatch.setattr(cw_probe, "_cache_file", lambda: cache)
        self.cache_path = cache

    def test_roundtrip(self):
        assert load_cached_context_window("m1", "http://x") is None
        save_cached_context_window("m1", "http://x", 131_072, "probed")
        assert load_cached_context_window("m1", "http://x") == 131_072
        # Different base_url or model: separate keys.
        assert load_cached_context_window("m1", None) is None
        assert load_cached_context_window("m2", "http://x") is None

    def test_entry_records_method(self):
        save_cached_context_window("m1", None, 65_536, "probed")
        data = json.loads(self.cache_path.read_text())
        entry = data["default|m1"]
        assert entry["context_window"] == 65_536
        assert entry["method"] == "probed"
        assert "detected_at" in entry

    def test_corrupt_cache_ignored(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text("{not json")
        assert load_cached_context_window("m1", None) is None
        # And writing over a corrupt file works.
        save_cached_context_window("m1", None, 32_768, "probed")
        assert load_cached_context_window("m1", None) == 32_768


# ---------------------------------------------------------------------------
# Resolution chain in LiteLLMBackend.get_model_info
# ---------------------------------------------------------------------------


class TestResolutionChain:
    @pytest.fixture(autouse=True)
    def tmp_cache(self, tmp_path, monkeypatch):
        cache = tmp_path / "context_windows.json"
        monkeypatch.setattr(cw_probe, "_cache_file", lambda: cache)

    def test_override_wins(self):
        p = LiteLLMBackend(model="unknown-xyz-model", context_window=99_000)
        info = p.get_model_info()
        assert info.context_window == 99_000
        assert info.cw_source == "override"

    def test_registry_hit(self):
        p = LiteLLMBackend(model="gpt-4o")
        info = p.get_model_info()
        assert info.cw_source == "registry"
        assert info.context_window >= 100_000

    def test_known_table_hit(self):
        # Not in litellm's registry under this exact name, but matches
        # the built-in table by substring.
        p = LiteLLMBackend(model="myserver/llama3.1-custom")
        info = p.get_model_info()
        assert info.cw_source == "known_table"
        assert info.context_window == 128_000

    def test_fallback_is_conservative_and_warns_once(self, caplog):
        p = LiteLLMBackend(model="totally-unknown-xyz")
        with caplog.at_level(logging.WARNING):
            info1 = p.get_model_info()
            info2 = p.get_model_info()
        assert info1.context_window == 32_768
        assert info1.cw_source == "fallback"
        assert info2.cw_source == "fallback"
        warnings = [r for r in caplog.records if "UNKNOWN" in r.message]
        assert len(warnings) == 1  # deduped

    def test_cache_rung_before_fallback(self):
        save_cached_context_window("totally-unknown-xyz", None, 131_072, "probed")
        p = LiteLLMBackend(model="totally-unknown-xyz")
        info = p.get_model_info()
        assert info.context_window == 131_072
        assert info.cw_source == "cache"

    @pytest.mark.asyncio
    async def test_probe_and_cache_integration(self, monkeypatch):
        async def fake_probe(model, **kwargs):
            return ProbeResult(65_536, "probed")

        monkeypatch.setattr(cw_probe, "probe_context_window", fake_probe)
        p = LiteLLMBackend(model="totally-unknown-xyz")
        assert p.get_model_info().cw_source == "fallback"

        detected = await p.probe_and_cache_context_window()
        assert detected == 65_536
        # Next resolution reads the cache.
        info = p.get_model_info()
        assert info.context_window == 65_536
        assert info.cw_source == "cache"
        # Second probe attempt is a no-op.
        assert await p.probe_and_cache_context_window() is None

    @pytest.mark.asyncio
    async def test_failed_probe_keeps_fallback(self, monkeypatch):
        async def fake_probe(model, **kwargs):
            return ProbeResult(None, "ceiling_distrust", "silent truncator")

        monkeypatch.setattr(cw_probe, "probe_context_window", fake_probe)
        p = LiteLLMBackend(model="totally-unknown-xyz")
        assert await p.probe_and_cache_context_window() is None
        assert p.get_model_info().cw_source == "fallback"


# ---------------------------------------------------------------------------
# Ollama /api/show parsing
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class TestOllamaShow:
    def _backend(self):
        return LiteLLMBackend(
            model="ollama/llama-custom", api_base="http://localhost:11434/v1"
        )

    def test_context_length_from_model_info(self, monkeypatch):
        def fake_get(url, timeout):
            return _FakeResponse(404, {})  # /v1/models rungs miss

        def fake_post(url, json, timeout):
            assert url.endswith("/api/show")
            assert json == {"model": "llama-custom"}
            return _FakeResponse(200, {
                "model_info": {"llama.context_length": 131_072},
                "parameters": "",
            })

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)
        assert self._backend()._query_server_context_window(
            "ollama/llama-custom"
        ) == 131_072

    def test_num_ctx_caps_model_max(self, monkeypatch):
        # Served context (num_ctx) below the model max: the real limit.
        def fake_get(url, timeout):
            return _FakeResponse(404, {})

        def fake_post(url, json, timeout):
            return _FakeResponse(200, {
                "model_info": {"llama.context_length": 131_072},
                "parameters": "num_ctx 4096\nstop \"<|end|>\"",
            })

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)
        assert self._backend()._query_server_context_window(
            "ollama/llama-custom"
        ) == 4_096


# ---------------------------------------------------------------------------
# llama.cpp /props parsing
# ---------------------------------------------------------------------------


class TestLlamaCppProps:
    def _backend(self):
        return LiteLLMBackend(
            model="openai/local-gguf", api_base="http://localhost:8080/v1"
        )

    def test_props_n_ctx_wins_over_trained_max(self, monkeypatch):
        """The serving n_ctx (30k slot) must be returned, never the
        trained max from /v1/models meta (262k)."""
        def fake_get(url, timeout):
            if url.endswith("/v1/models"):
                return _FakeResponse(200, {
                    "data": [{
                        "id": "local-gguf",
                        "meta": {"n_ctx_train": 262_144, "n_vocab": 151_936},
                    }],
                })
            if url.endswith("/props"):
                return _FakeResponse(200, {
                    "default_generation_settings": {"id": 0, "n_ctx": 30_000},
                    "total_slots": 3,
                })
            return _FakeResponse(404, {})

        def fake_post(url, json, timeout):
            return _FakeResponse(404, {})

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)
        assert self._backend()._query_server_context_window(
            "openai/local-gguf"
        ) == 30_000

    def test_props_absent_falls_through(self, monkeypatch):
        def fake_get(url, timeout):
            return _FakeResponse(404, {})

        def fake_post(url, json, timeout):
            return _FakeResponse(404, {})

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr("requests.post", fake_post)
        assert self._backend()._query_server_context_window(
            "openai/local-gguf"
        ) is None


# ---------------------------------------------------------------------------
# context_window setting -> backend constructor plumbing
# ---------------------------------------------------------------------------


class TestSettingsPlumbing:
    def test_settings_context_window_reaches_backend(self, caplog):
        from alancode.agent import _create_backend_from_settings

        backend = _create_backend_from_settings({
            "backend": "auto",
            "model": "totally-unknown-xyz",
            "context_window": 30_000,
        })
        with caplog.at_level(logging.WARNING):
            info = backend.get_model_info()
        assert info.context_window == 30_000
        assert info.cw_source == "override"
        assert not [r for r in caplog.records if "UNKNOWN" in r.message]

    def test_settings_auto_leaves_override_unset(self):
        from alancode.agent import _create_backend_from_settings

        backend = _create_backend_from_settings({
            "backend": "auto",
            "model": "totally-unknown-xyz",
            "context_window": "auto",
        })
        assert backend._context_window_override is None

    def test_anthropic_native_ignores_context_window(self):
        from alancode.agent import _create_backend_from_settings

        backend = _create_backend_from_settings({
            "backend": "anthropic-native",
            "model": "claude-sonnet-4-6",
            "api_key": "test-key",
            "context_window": 30_000,
        })
        assert backend is not None
