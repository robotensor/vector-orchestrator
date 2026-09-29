"""Building small published histories for the store tests and the fixture store, and checking
that what a store holds is what the dashboard can read."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vector_orchestrator.canon import canonical_json, sha256_file, sha256_hex
from vector_orchestrator.ids import SubmissionRef, event_id
from vector_orchestrator.spec import load_schema, load_spec_file
from vector_orchestrator.store.records import (
    duel_event,
    empty_skill_scores,
    index_record,
    media_shas,
    unit_tally,
)
from vector_orchestrator.store.writer import Store, king_after

TRACK = "franka_1arm"
FINISHED = "2026-09-13T12:00:00Z"
STARTED = "2026-09-13T11:00:00Z"


def small_spec(spec, tmp_path: Path, lines_per_part: int = 2):
    doc = copy.deepcopy(spec.raw)
    doc["store"]["index_lines_per_part"] = lines_per_part
    path = tmp_path / f"spec-{lines_per_part}.json"
    path.write_text(json.dumps(doc))
    return load_spec_file(path)


def make_record(
    spec,
    kind: str,
    block: int,
    king: SubmissionRef | None,
    challenger: SubmissionRef | None,
    *,
    dethroned: bool = False,
    track: str = TRACK,
    finished_at: str = FINISHED,
) -> dict:
    subject = (challenger or king).key  # type: ignore[union-attr]
    skill = spec.skills(track)[0]
    king_scores = challenger_scores = None
    if kind == "duel":
        king_scores = {**empty_skill_scores(spec.skills(track)), skill: 0.5, "average": 0.5}
        challenger_scores = {**empty_skill_scores(spec.skills(track)), skill: 0.9, "average": 0.9}
    return index_record(
        schema=spec.store["schema"],
        event_id=event_id(kind, track, block, subject),
        kind=kind,
        track=track,
        block=block,
        finished_at=finished_at,
        king=king,
        challenger=challenger,
        king_scores=king_scores,
        challenger_scores=challenger_scores,
        score_margin=spec.score_margin(track),
        dethroned=dethroned,
        new_king=challenger if dethroned else None,
        duel_size="smoke",
    )


def publish(
    store: Store, spec, record: dict, units=None, track: str = TRACK, started_at: str = STARTED
) -> int:
    units = units or []
    record.update(unit_tally(units), media_count=len(media_shas(units)))
    event = duel_event(
        record,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        units=units,
        units_per_skill=spec.units_per_skill(track, "smoke"),
        started_at=started_at,
        wall_seconds=1.5,
        demonstration=spec.demonstration(track),
    )
    store.write_event(track, event)
    return store.append(track, record)


def schema_errors(ref: str, obj: Any) -> list[str]:
    """Where `obj` breaks `$defs/{ref}` of store-schema.json; empty when it is that shape."""
    import jsonschema

    defs = load_schema()["$defs"]
    validator = jsonschema.Draft202012Validator({"$ref": f"#/$defs/{ref}", "$defs": defs})
    return [
        f"{ref}: {e.message} at {'/'.join(str(p) for p in e.path)}"
        for e in validator.iter_errors(obj)
    ]


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
    """What the tests hold a written store to: every index line whole canonical JSON, seqs
    contiguous from 1, every event the bytes its record's `event_sha256` names and repeating the
    record, every clip present under its hash, each head the index's last record and the king its
    records crown, and every file the shape store-schema.json gives it."""
    store = Store(root, spec)
    out = Checked()
    manifest = store.manifest()
    if manifest is None:
        out.errors.append("manifest.json missing or unreadable")
        return out
    out.errors += schema_errors("Manifest", manifest)
    for track in spec.tracks:
        seq, last, king = 0, None, None
        for part in store.index_parts(track):
            path = store.index_part_path(track, part)
            data = path.read_bytes()
            if data and not data.endswith(b"\n"):
                out.errors.append(f"{path.name}: the last line has no newline")
            for n, raw in enumerate(data.split(b"\n")[:-1], start=1):
                where = f"{track}/{path.name}:{n}"
                body = raw.split(b"\t", 1)[0].decode("utf-8")
                record = json.loads(body)
                if canonical_json(record) != body:
                    out.errors.append(f"{where}: not canonical JSON")
                seq += 1
                if record.get("seq") != seq or store.part_of(seq) != part:
                    out.errors.append(f"{where}: seq {record.get('seq')}, expected {seq}")
                out.errors += [f"{where}: {e}" for e in schema_errors("IndexRecord", record)]
                out.records += 1
                last, king = record, king_after(record, king)
                event_bytes = store.event_path(track, record["event_id"]).read_bytes()
                if record.get("event_sha256") != sha256_hex(event_bytes):
                    out.errors.append(f"{where}: the event is not the bytes event_sha256 names")
                event = json.loads(event_bytes)
                out.events += 1
                out.errors += [f"{where}: {e}" for e in schema_errors("DuelEvent", event)]
                for k in sorted(set(record) - {"seq", "event_sha256"}):
                    if event.get(k) != record[k]:
                        out.errors.append(f"{where}: event.{k} differs from the record")
                for sha in media_shas(event.get("units") or []):
                    out.media += 1
                    clip = store.media_path(sha, spec.video_format)
                    if not clip.is_file() or sha256_file(clip) != sha:
                        out.errors.append(f"{where}: media {sha[:12]} missing or not its hash")
        head = store.head(track) or {}
        out.errors += [f"{track} head: {e}" for e in schema_errors("Head", head)]
        if head.get("seq") != (last["seq"] if last else 0) or head.get("king") != king:
            out.errors.append(f"{track} head is not the index's last record and king")
        if store.queue_path(track).exists():
            snapshot = json.loads(store.queue_path(track).read_text())
            out.errors += [f"{track} queue: {e}" for e in schema_errors("QueueSnapshot", snapshot)]
    return out
