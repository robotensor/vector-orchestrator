"""Building small published histories for the store tests and the fixture store, and checking that
what a store holds is exactly the published shape the dashboard reads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from vector_orchestrator.canon import canonical_json, sha256_file
from vector_orchestrator.ids import SubmissionRef, event_id
from vector_orchestrator.store.records import index_record, media_shas, unit_row
from vector_orchestrator.store.writer import SCHEMA, Store, king_after

FINISHED = "2026-09-13T12:00:00Z"
STARTED = "2026-09-13T11:00:00Z"

RECORD_KEYS = {
    "seq", "event_id", "kind", "block", "finished_at", "king", "challenger", "king_score",
    "challenger_score", "dethroned", "wins", "losses", "ties", "void",
}  # fmt: skip
EVENT_KEYS = {
    "event_id", "duel_id", "kind", "block", "seed", "spec_version", "runs_per_task", "started_at",
    "finished_at", "king", "challenger", "scores", "dethroned", "verdict", "benchmark", "notes",
    "units",
}  # fmt: skip
UNIT_KEYS = set(unit_row({"unit_id": "x-00", "task": "x"}))
HEAD_KEYS = {"seq", "block", "king"}


def make_record(
    kind: str,
    block: int,
    king: SubmissionRef,
    challenger: SubmissionRef | None,
    *,
    dethroned: bool = False,
    finished_at: str = FINISHED,
    units: list | None = None,
) -> dict:
    subject = (challenger or king).key
    return index_record(
        event_id=event_id(kind, block, subject),
        kind=kind,
        block=block,
        finished_at=finished_at,
        king=king.as_dict(),
        challenger=challenger.as_dict() if challenger else None,
        king_score=0.5,
        challenger_score=0.9 if challenger else None,
        dethroned=dethroned,
        units=units or [],
    )


def publish(store: Store, spec, record: dict, units=None, started_at: str = STARTED) -> int:
    units = units or []
    event = {
        "event_id": record["event_id"],
        "duel_id": "d" * 64,
        "kind": record["kind"],
        "block": record["block"],
        "seed": None,
        "spec_version": spec.version,
        "runs_per_task": spec.runs_per_task,
        "started_at": started_at,
        "finished_at": record["finished_at"],
        "king": {**record["king"], "weights_sha256": None, "refused": None},
        "challenger": {**record["challenger"], "weights_sha256": None, "refused": None}
        if record["challenger"]
        else None,
        "scores": {"king": {"average": record["king_score"]}, "challenger": None},
        "dethroned": record["dethroned"],
        "verdict": {"reason": "genesis" if record["kind"] == "genesis" else "margin-met"},
        "benchmark": {"commit": "fake", "source_sha256": "fake"},
        "notes": [],
        "units": units,
    }
    store.write_event(event)
    return store.append(record)


@dataclass
class Checked:
    errors: list[str] = field(default_factory=list)
    records: int = 0
    events: int = 0
    media: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def check_store(root: str | Path, spec) -> Checked:
    """Every index line a record's canonical JSON with exactly `RECORD_KEYS`, seqs contiguous from
    1, every event present with exactly `EVENT_KEYS` and agreeing with its record, every unit with
    exactly `UNIT_KEYS`, every clip present under its hash, and the head the index's last record
    and the king its records crown."""
    store = Store(root, spec)
    out = Checked()
    manifest = store.manifest()
    if manifest != {"schema": SCHEMA}:
        out.errors.append(f"manifest.json is {manifest}")
    seq, last, king = 0, None, None
    for part in store.index_parts():
        path = store.index_part_path(part)
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            out.errors.append(f"{path.name}: the last line has no newline")
        for n, raw in enumerate(data.split(b"\n")[:-1], start=1):
            where = f"{path.name}:{n}"
            try:
                record = json.loads(raw.decode("utf-8"))
            except ValueError:
                out.errors.append(f"{where}: not JSON")
                continue
            if canonical_json(record) != raw.decode("utf-8"):
                out.errors.append(f"{where}: not canonical JSON")
            if set(record) != RECORD_KEYS:
                out.errors.append(f"{where}: keys {sorted(set(record) ^ RECORD_KEYS)} differ")
            seq += 1
            if record.get("seq") != seq or store.part_of(seq) != part:
                out.errors.append(f"{where}: seq {record.get('seq')}, expected {seq}")
            out.records += 1
            last, king = record, king_after(record, king)
            event = store.event(record["event_id"])
            if event is None:
                out.errors.append(f"{where}: no event")
                continue
            out.events += 1
            if set(event) != EVENT_KEYS:
                out.errors.append(f"{where}: event keys {sorted(set(event) ^ EVENT_KEYS)} differ")
            for k in ("event_id", "kind", "block", "finished_at", "dethroned"):
                if event.get(k) != record[k]:
                    out.errors.append(f"{where}: event.{k} differs from the record")
            for unit in event.get("units") or []:
                if set(unit) != UNIT_KEYS:
                    out.errors.append(f"{where}: unit keys {sorted(set(unit) ^ UNIT_KEYS)} differ")
            for sha in media_shas(event.get("units") or []):
                out.media += 1
                clip = store.media_path(sha)
                if not clip.is_file() or sha256_file(clip) != sha:
                    out.errors.append(f"{where}: media {sha[:12]} missing or not its hash")
    head = store.head() or {}
    if set(head) != HEAD_KEYS:
        out.errors.append(f"head keys {sorted(head)}")
    if head.get("seq") != (last["seq"] if last else 0) or head.get("king") != king:
        out.errors.append("the head is not the index's last record and king")
    return out
