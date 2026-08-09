"""Endpoint discovery.

Foundry Local binds a dynamic port, so the endpoint is found at runtime. The bug these
cover: discovery used to be lru_cached, so with the server stopped the first call fell
back to the documented default and that answer stuck for the life of the process.
Starting Foundry afterwards changed nothing -- every later request reported "cannot
reach 5267" while the server sat on another port -- and only restarting the app helped.
"""

from __future__ import annotations

import pytest

from malimusavir import foundry


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Discovery memoises across calls; each test starts from nothing."""
    monkeypatch.setattr(foundry, "_ENDPOINT", None)
    monkeypatch.delenv("FOUNDRY_LOCAL_ENDPOINT", raising=False)
    foundry._client_for.cache_clear()
    foundry._model_ids_at.cache_clear()
    yield
    monkeypatch.setattr(foundry, "_ENDPOINT", None)


def test_a_successful_probe_is_remembered(monkeypatch):
    calls = []

    def probe():
        calls.append(1)
        return "http://127.0.0.1:5648"

    monkeypatch.setattr(foundry, "_probe_endpoint", probe)
    assert foundry.discover_endpoint() == "http://127.0.0.1:5648"
    assert foundry.discover_endpoint() == "http://127.0.0.1:5648"
    assert len(calls) == 1, "a working endpoint should not be re-probed every call"


def test_a_failed_probe_is_not_remembered(monkeypatch):
    """The whole point. A fallback that sticks makes the app unrecoverable without a
    restart, which is exactly what happened."""
    calls = []

    def probe():
        calls.append(1)
        return None

    monkeypatch.setattr(foundry, "_probe_endpoint", probe)
    assert foundry.discover_endpoint() == foundry.DEFAULT_ENDPOINT
    assert foundry.discover_endpoint() == foundry.DEFAULT_ENDPOINT
    assert len(calls) == 2, "a failed probe must be retried, not cached"


def test_the_app_recovers_when_the_server_starts_later(monkeypatch):
    """Stopped at first call, running at the second -- no restart in between."""
    state = {"running": False}
    monkeypatch.setattr(
        foundry, "_probe_endpoint",
        lambda: "http://127.0.0.1:5648" if state["running"] else None)

    assert foundry.discover_endpoint() == foundry.DEFAULT_ENDPOINT
    state["running"] = True
    assert foundry.discover_endpoint() == "http://127.0.0.1:5648"


def test_an_explicit_override_wins_and_is_never_probed(monkeypatch):
    monkeypatch.setenv("FOUNDRY_LOCAL_ENDPOINT", "http://example.invalid:9000/")
    monkeypatch.setattr(foundry, "_probe_endpoint",
                        lambda: pytest.fail("must not probe when overridden"))
    assert foundry.discover_endpoint() == "http://example.invalid:9000"


def test_the_client_follows_a_moved_endpoint(monkeypatch):
    """A client cached outright would keep talking to the old port."""
    monkeypatch.setattr(foundry, "_probe_endpoint", lambda: "http://127.0.0.1:1111")
    first = foundry.get_client()
    monkeypatch.setattr(foundry, "_ENDPOINT", None)
    monkeypatch.setattr(foundry, "_probe_endpoint", lambda: "http://127.0.0.1:2222")
    assert foundry.get_client() is not first


def test_the_error_names_the_endpoint_it_actually_tried(monkeypatch):
    monkeypatch.setattr(foundry, "_probe_endpoint", lambda: None)
    monkeypatch.setattr(foundry, "_model_ids_at",
                        lambda base: (_ for _ in ()).throw(RuntimeError("refused")))
    with pytest.raises(foundry.FoundryError) as caught:
        foundry._loaded_model_ids()
    assert foundry.DEFAULT_ENDPOINT in str(caught.value)
    assert "foundry server start" in str(caught.value)


def test_model_lookup_reprobes_before_giving_up(monkeypatch):
    """The server may have come up since the last call; one retry turns "restart the
    app" into "it just works"."""
    state = {"running": False}
    monkeypatch.setattr(
        foundry, "_probe_endpoint",
        lambda: "http://127.0.0.1:5648" if state["running"] else None)

    def ids(base):
        if base == "http://127.0.0.1:5648":
            return ("qwen3-4b-generic-cpu",)
        state["running"] = True          # comes up between the two attempts
        raise RuntimeError("connection refused")

    monkeypatch.setattr(foundry, "_model_ids_at", ids)
    assert foundry._loaded_model_ids() == ("qwen3-4b-generic-cpu",)


# --- starting the server ----------------------------------------------------------------


def test_ensure_server_does_not_start_one_that_is_already_up(monkeypatch):
    monkeypatch.setattr(foundry, "_probe_endpoint", lambda: "http://127.0.0.1:5648")
    monkeypatch.setattr(foundry.subprocess, "run",
                        lambda *a, **k: pytest.fail("must not start a running server"))
    assert foundry.ensure_server() == "http://127.0.0.1:5648"


def test_ensure_server_returns_none_when_the_cli_is_missing(monkeypatch):
    """Foundry not being installed must degrade the assistant, not stop the app: the
    ledger, the deadline board and the export are all pure SQL."""
    monkeypatch.setattr(foundry, "_probe_endpoint", lambda: None)

    def missing(*args, **kwargs):
        raise FileNotFoundError("foundry")

    monkeypatch.setattr(foundry.subprocess, "run", missing)
    assert foundry.ensure_server() is None


def test_ensure_server_reports_the_endpoint_after_starting(monkeypatch):
    state = {"running": False}
    monkeypatch.setattr(
        foundry, "_probe_endpoint",
        lambda: "http://127.0.0.1:5648" if state["running"] else None)

    def start(*args, **kwargs):
        state["running"] = True
        return None

    monkeypatch.setattr(foundry.subprocess, "run", start)
    assert foundry.ensure_server() == "http://127.0.0.1:5648"
