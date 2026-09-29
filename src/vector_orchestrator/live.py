"""Live progress frames, POSTed to the dashboard's `/api/live`: ephemeral, best-effort, never
part of the record.

    {schema, event_id, kind, king, challenger, phase, side, current, message, started_at, sent_at,
     units: [{unit_id, task, void, outcome, king_success, challenger_success,
              demo_video, king_video, challenger_video}]}

The dashboard checks a frame strictly where a wrong value would misfile it, so `build_frame`
refuses to build one it would refuse.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import socket
import threading
import time
import urllib.parse
from typing import Any

from .store.records import now_iso

log = logging.getLogger(__name__)

SCHEMA = 5
PATH = "/api/live"
MAX_FRAME_BYTES = 512 << 10
MIN_INTERVAL_S = 1.0
#: The phases of a duel, in order (`PHASES` in the dashboard's live types must equal this).
PHASES = ("fetching", "checking", "materializing", "evaluating", "publishing", "done", "failed")
SIDES = ("challenger", "king")
EVENT_ID_RE = re.compile(r"[0-9a-f]{8,64}")
MAX_MESSAGE = 300
UNIT_FIELDS = (
    "unit_id", "task", "void", "outcome", "king_success", "challenger_success",
    "demo_video", "king_video", "challenger_video",
)  # fmt: skip


def build_frame(
    *,
    event_id: str,
    kind: str,
    king: dict[str, str] | None,
    challenger: dict[str, str] | None,
    phase: str,
    side: str | None,
    units: list[dict[str, Any]],
    current: dict[str, Any] | None,
    message: str,
    started_at: str,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    if not EVENT_ID_RE.fullmatch(event_id):
        raise ValueError(f"event_id must be 8-64 lowercase hex characters, not {event_id!r}")
    if side is not None and side not in SIDES:
        raise ValueError(f"side must be one of {SIDES} or None, not {side!r}")
    for u in units:
        if not isinstance(u.get("task"), str) or not u["task"]:
            raise ValueError(f"unit {u.get('unit_id')}: task must be a non-empty string")
    return {
        "schema": SCHEMA,
        "event_id": event_id,
        "kind": kind,
        "king": king,
        "challenger": challenger,
        "phase": phase,
        "side": side,
        "current": current,
        "message": message[:MAX_MESSAGE],
        "started_at": started_at,
        "sent_at": now_iso(),
        "units": [{k: u.get(k) for k in UNIT_FIELDS} for u in units],
    }


#: Of the dashboard's answer, which is a status and nothing the orchestrator acts on.
MAX_RESPONSE_BYTES = 4096


class LiveReporter:
    """POSTs frames to the dashboard. Failures are logged and otherwise ignored: a frame is a
    window onto a duel, and a duel must never stop because the window is shut.

    `timeout_s` bounds the whole request, not each socket operation: a dashboard that answers a
    byte at a time would otherwise hold a duel for as long as it liked. A redirect is a failed
    delivery and is never followed - the bearer token opens the ingest, and must not travel to
    whatever host an answer names.
    """

    def __init__(self, url: str | None, token: str | None, *, timeout_s: float = 5.0):
        self.url = (url.rstrip("/") + PATH) if url else None
        self.token = token
        self.timeout_s = timeout_s
        self.min_interval_s = MIN_INTERVAL_S
        self.max_bytes = MAX_FRAME_BYTES
        self._last_sent = 0.0
        self.sent = 0
        self.failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.token)

    def encode(self, frame: dict[str, Any]) -> bytes:
        """The body to POST: the frame, or the frame without its unit list if it is too large."""
        data = json.dumps(frame).encode("utf-8")
        if len(data) > self.max_bytes:
            data = json.dumps({**frame, "units": []}).encode("utf-8")
        return data

    def _deliver(self, body: bytes) -> None:
        """POST `body` within `timeout_s`, or raise. The request runs on a worker thread whose
        socket is shut from under it when the deadline passes, so no call can outlast the frame."""
        parts = urllib.parse.urlsplit(str(self.url))
        connect = (
            http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        )
        conn = connect(parts.netloc, timeout=self.timeout_s)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        outcome: dict[str, Any] = {}

        def send() -> None:
            try:
                conn.request(
                    "POST",
                    path,
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.token}",
                    },
                )
                response = conn.getresponse()
                response.read(MAX_RESPONSE_BYTES)
                outcome["status"] = response.status
            except Exception as exc:  # noqa: BLE001 - re-raised on the calling thread
                outcome["error"] = exc

        worker = threading.Thread(target=send, daemon=True)
        worker.start()
        worker.join(self.timeout_s)
        try:
            if worker.is_alive():
                sock = conn.sock
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                worker.join(1.0)
                raise TimeoutError(f"no answer within {self.timeout_s:g}s")
            if "error" in outcome:
                raise outcome["error"]
            status = int(outcome.get("status", 0))
            if 300 <= status < 400:
                raise http.client.HTTPException(f"{status}: redirects are not followed")
            if not 200 <= status < 300:
                raise http.client.HTTPException(f"the dashboard answered {status}")
        finally:
            conn.close()

    def post(self, frame: dict[str, Any], *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        if not force and now - self._last_sent < self.min_interval_s:
            return False
        try:
            self._deliver(self.encode(frame))
        except Exception as exc:  # noqa: BLE001 - a shut window never stops a duel
            self.failed += 1
            log.warning("live frame not delivered: %s", exc)
            return False
        self._last_sent = now
        self.sent += 1
        return True
