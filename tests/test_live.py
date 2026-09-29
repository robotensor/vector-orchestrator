"""Live frames: what `build_frame` builds and refuses, and how a frame reaches the dashboard."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vector_orchestrator.live import (
    MAX_FRAME_BYTES,
    PATH,
    PHASES,
    UNIT_FIELDS,
    LiveReporter,
    build_frame,
)
from vector_orchestrator.store.records import unit_row
from vector_orchestrator.units import deal


def units(spec):
    rows = [unit_row(u) for u in deal(spec, "e" * 64, None, 1)]
    rows[0].update(king_success=True, challenger_success=False, outcome="king", king_video="a" * 64)
    rows[1].update(void=True)
    return rows


def frame(spec=None, **overrides):
    from vector_orchestrator.spec import load_spec

    spec = spec or load_spec()
    args = dict(
        event_id="e" * 64,
        kind="duel",
        king={"key": "k" * 16, "repo": "org/king", "revision": "a" * 40},
        challenger={"key": "c" * 16, "repo": "org/challenger", "revision": "b" * 40},
        phase="evaluating",
        side="challenger",
        units=units(spec),
        current={"unit_id": "beat_block_hammer-00", "side": "challenger"},
        message="challenger: beat_block_hammer-00",
        started_at="2026-09-13T10:00:00Z",
    )
    args.update(overrides)
    return build_frame(**args)


def test_a_frame_carries_the_duel_and_every_unit_row(spec):
    built = frame(spec)
    assert set(built) == {
        "schema", "event_id", "kind", "king", "challenger", "phase", "side", "current",
        "message", "started_at", "sent_at", "units",
    }  # fmt: skip
    assert len(built["units"]) == len(spec.tasks)
    assert set(built["units"][0]) == set(UNIT_FIELDS)
    assert built["units"][0]["king_video"] == "a" * 64 and built["units"][1]["void"] is True
    assert PHASES.index("materializing") == PHASES.index("checking") + 1
    for phase in PHASES:
        assert frame(spec, phase=phase, side=None)["phase"] == phase


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"phase": "sleeping"}, "phase"),
        ({"event_id": "E" * 64}, "event_id"),
        ({"event_id": "e" * 64 + "\n"}, "event_id"),
        ({"side": "referee"}, "side"),
    ],
)
def test_a_frame_the_dashboard_would_refuse_is_not_built(spec, overrides, message):
    with pytest.raises(ValueError, match=message):
        frame(spec, **overrides)
    rows = units(spec)
    rows[0]["task"] = ""
    with pytest.raises(ValueError, match="task"):
        frame(spec, units=rows)


def test_a_large_unit_list_is_slimmed_to_fit_rather_than_refused(spec):
    big = frame(spec, units=units(spec) * 400)
    body = LiveReporter(None, None).encode(big)
    assert len(body) <= MAX_FRAME_BYTES and json.loads(body)["units"] == []
    assert json.loads(LiveReporter(None, None).encode(frame(spec)))["units"]


class _Sink(BaseHTTPRequestHandler):
    frames: list = []

    def log_message(self, *a):
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Sink.frames.append(
            (self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(n)))
        )
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_GET(self):
        _Sink.frames.append((self.path, self.headers.get("Authorization"), {}))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


class _Slow(BaseHTTPRequestHandler):
    """Answers a byte at a time, for longer than any duel would wait."""

    def log_message(self, *a):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Length", "20")
        self.end_headers()
        for _ in range(20):
            try:
                self.wfile.write(b"x")
                self.wfile.flush()
            except OSError:
                return
            time.sleep(0.5)


class _Redirect(BaseHTTPRequestHandler):
    elsewhere = ""

    def log_message(self, *a):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(302)
        self.send_header("Location", _Redirect.elsewhere)
        self.send_header("Content-Length", "0")
        self.end_headers()


@contextmanager
def serving(handler):
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def test_a_slow_dashboard_does_not_hold_the_duel(spec):
    """The timeout is the whole request's, not each socket read's: a window that never finishes
    answering must not stop a duel."""
    with serving(_Slow) as url:
        rep = LiveReporter(url, "tok", timeout_s=1.0)
        started = time.monotonic()
        assert not rep.post(frame(), force=True)
        assert time.monotonic() - started < 3, "the reporter waited on a slow answer"
        assert rep.failed == 1


def test_the_bearer_token_never_follows_a_redirect(spec):
    """urllib would replay the POST as a GET and carry the Authorization header to whatever host
    the answer names; the live token opens the dashboard's ingest."""
    _Sink.frames = []
    with serving(_Sink) as elsewhere, serving(_Redirect) as url:
        _Redirect.elsewhere = f"{elsewhere}/steal"
        rep = LiveReporter(url, "SECRET-TOKEN", timeout_s=2.0)
        assert not rep.post(frame(), force=True) and rep.failed == 1
    assert _Sink.frames == [], "the token was sent to another host"


def test_the_reporter_posts_with_a_bearer_token_and_never_raises(spec):
    built = frame()
    _Sink.frames = []
    httpd = HTTPServer(("127.0.0.1", 0), _Sink)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        rep = LiveReporter(f"http://127.0.0.1:{httpd.server_address[1]}", "tok")
        assert rep.enabled and rep.post(built, force=True)
        assert not rep.post(built)  # rate limited
        assert rep.post(built, force=True)
        path, auth, got = _Sink.frames[0]
        assert path == PATH and auth == "Bearer tok" and got["event_id"] == "e" * 64
        off = LiveReporter(None, None)
        assert not off.enabled and not off.post(built, force=True)
        dead = LiveReporter("http://127.0.0.1:9", "tok", timeout_s=0.5)
        assert not dead.post(built, force=True) and dead.failed == 1
    finally:
        httpd.shutdown()
