"""A duel from request to published record.

    fetching -> checking -> materializing + evaluating(challenger) + evaluating(king)
             -> publishing -> done | failed

**Where it runs.** Once both sides are checked, the demonstrations and both sides' units run
together in the duel's `Slots`: `workers` slots on each GPU of the host (`slots.gpus`), a job in a
slot at a time, on that slot's card. A unit is played by both sides as soon as its prompt lands,
before new demonstrations are made, and each slot keeps its policy server between units.

**Identity.** A duel's id is `ids.duel_id(spec.version, challenger, king)` and its units are
`units.deal(spec, duel_id, entropy)`, pure functions of those, so anyone holding the record can
deal them again. Its event id adds the block. Everything the duel keeps on this host lives in
`<run_root>/<event_id[:16]>/`:

    request.json                  what was asked, and when it started
    prompts/<unit_id>/prompt.npz  every unit's prompt, and prompts/manifest.jsonl
    <side>/results.jsonl          each side's finished units, and <side>/<unit_id>/ its run logs
    forfeit.json                  why the king forfeits, when it does
    outcome.json                  what was decided: published (with the record), refused or void
    failed.txt                    why the last attempt stopped, for a duel that did not finish

**Resuming.** Every stage is resumable from that directory (an interrupt ends what the slots were
running and records none of it): prompts are re-used after their hash
is checked, a side runs only the units its results file does not hold, and a duel first looks for
its own event id in the index, so a duel killed at any point and run again finishes with each unit
run once per side and at most one record. One found there is not run again: its head is rebuilt
from the index (a kill between the index line and the head's rewrite leaves the old king in it)
and its files are pushed to the mirror again. Once `outcome.json` exists, running the duel again
returns it.

**What stops a duel, and what that means.**

- `DuelFailed`: the harness could not run it (no benchmark, the Hub unreachable, a bug). Nothing
  is published and nothing is decided; running it again resumes it. Its subclass
  `HarnessUnavailable` is the part of that which may pass by itself - the checkout missing or
  unusable, the Hub unreachable - which the caller retries later.
- Refused: the challenger's submission cannot run (a repository holding other files, weights that
  fail the check or do not load). Nothing else runs - the king is not fetched and no prompt is made - nothing is
  published, and `outcome.json` says why.
- Void: the duel ran and cannot stand. The crown does not move, nothing is published, and
  `outcome.json` says why. A duel is void when more than `crown.max_void` of its units are void
  (asked before every job, so nothing more is run once it is certain). A unit is void only for a
  harness cause (`side`); what a side's own submission does to a unit is that side's failure.
- A king whose submission is refused (its repository gone or private) forfeits: every one of its units is a failure, the duel is
  scored and published as any other with the note `king forfeit: <reason>`, and the challenger
  takes the crown if its own average clears the margin. The forfeit is recorded in
  `forfeit.json`, so a resumed duel keeps it.
- `CrownMoved`: the head no longer names the king the duel was asked against (someone
  else was crowned while this duel was stopped). Checked before the duel starts or resumes, and
  again just before it publishes: its run directory is moved aside as `<dir>.stale-<n>`, nothing
  is published, and the caller queues the challenger again. A record against a king who has lost
  the crown would hand it back to him.
- A published duel moves the crown iff `score.crown_moves`.

**Genesis.** With no king yet, the first challenger is crowned by an event of kind `genesis`: its side is materialized and evaluated like any other,
and published with its own scores in the record's `king` slot, which is where a genesis names what
it crowns. A genesis is void on the same rule as a duel.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ids import SubmissionRef, duel_id, event_id
from ..progress import ProgressBoard, build_progress
from ..runner import RUNNING
from ..spec import Spec
from ..store.records import index_record, media_shas, now_iso, unit_row
from ..store.writer import Store, atomic_write_json, read_json
from ..units import deal
from . import score
from .materialize import Materialized, Prompt, materialize_units
from .orphans import move_aside, reap_run_root
from .runtime import PreparedSubmission, RuntimeUnavailable, SubmissionRefused
from .side import read_results, run_side
from .slots import Slots, gpus

log = logging.getLogger(__name__)

REQUEST_FILE = "request.json"
OUTCOME_FILE = "outcome.json"
FAILED_FILE = "failed.txt"
#: The king's refusal, recorded once: a resumed duel keeps the forfeit rather than asking again.
FORFEIT_FILE = "forfeit.json"
SIDES = ("challenger", "king")
#: Finished clips are put in the store (and mirrored) every this many units.
MEDIA_FLUSH_UNITS = 4


class DuelFailed(RuntimeError):
    """The harness could not run the duel. Nothing was published or decided."""


class HarnessUnavailable(DuelFailed):
    """A part of the harness the duel needs is not there for now - the checkout missing or
    unusable, the Hub unreachable. Worth trying the same duel again later."""


class CrownMoved(RuntimeError):
    """The crown is no longer held by the king the duel was asked against. Nothing was
    published, the duel's run was moved aside (`moved_to`), and its challenger is to be queued
    again: a record against a king who no longer holds the crown would hand it back to him."""

    def __init__(self, req: DuelRequest, reason: str, moved_to: Path | None) -> None:
        self.req = req
        self.reason = reason
        self.moved_to = moved_to
        super().__init__(reason)


@dataclass(frozen=True)
class DuelRequest:
    challenger: SubmissionRef
    #: None where there is no king: the duel is then a genesis.
    king: SubmissionRef | None
    #: The block this duel is fought in; part of its event id.
    block: int
    #: The chain block whose hash seeds the duel's units, and that hash as the chain gave it
    #: (`0x` + 64 hex). Both or neither: without them a duel's units are a function of public
    #: values alone, computable before anyone submits.
    seed_block: int | None = None
    seed_block_hash: str | None = None
    #: Units per task, when not the spec's (a manual duel's smaller run).
    runs_per_task: int | None = None

    def __post_init__(self) -> None:
        if (self.seed_block is None) != (self.seed_block_hash is None):
            raise ValueError("a duel request names a seed block and its hash, or neither")
        if self.seed_block is not None and (
            isinstance(self.seed_block, bool) or int(self.seed_block) < 0
        ):
            raise ValueError(f"seed_block {self.seed_block!r} is not a block number")
        if self.seed_block_hash is not None and not _is_block_hash(self.seed_block_hash):
            raise ValueError(f"seed_block_hash {self.seed_block_hash!r} is not 0x + 64 hex")

    @property
    def kind(self) -> str:
        return "duel" if self.king is not None else "genesis"

    @property
    def entropy(self) -> str | None:
        """What the units are seeded with besides the duel id: `<block>:<hash>`, or None."""
        if self.seed_block is None:
            return None
        return f"{int(self.seed_block)}:{str(self.seed_block_hash).lower()}"

    def duel_id(self, spec: Spec) -> str:
        return duel_id(spec.version, self.challenger, self.king)

    def event_id(self, spec: Spec) -> str:
        return event_id(self.kind, self.block, self.duel_id(spec))

    def runs(self, spec: Spec) -> int:
        return spec.runs_per_task if self.runs_per_task is None else int(self.runs_per_task)

    def as_dict(self, spec: Spec) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "challenger": self.challenger.as_dict(),
            "king": self.king.as_dict() if self.king else None,
            "runs_per_task": self.runs_per_task,
            "block": self.block,
            "duel_id": self.duel_id(spec),
            "event_id": self.event_id(spec),
            # Only when set, so a request written before seeds existed still matches itself.
            **(
                {"seed_block": int(self.seed_block), "seed_block_hash": self.seed_block_hash}
                if self.seed_block is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> DuelRequest:
        challenger = SubmissionRef.from_dict(doc["challenger"])
        if challenger is None:
            raise ValueError("a duel request names no challenger")
        return cls(
            challenger=challenger,
            king=SubmissionRef.from_dict(doc.get("king")),
            block=int(doc["block"]),
            seed_block=None if doc.get("seed_block") is None else int(doc["seed_block"]),
            seed_block_hash=doc.get("seed_block_hash"),
            runs_per_task=doc.get("runs_per_task"),
        )


def _is_block_hash(value: Any) -> bool:
    text = str(value)
    return (
        isinstance(value, str)
        and text.startswith("0x")
        and len(text) == 66
        and all(c in "0123456789abcdefABCDEF" for c in text[2:])
    )


@dataclass
class DuelResult:
    #: published or void.
    status: str
    kind: str
    event_id: str
    duel_id: str
    reason: str
    run_dir: Path
    record: dict[str, Any] | None = None
    units: list[dict[str, Any]] = field(default_factory=list)

    @property
    def published(self) -> bool:
        return self.status == "published"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.kind,
            "event_id": self.event_id,
            "duel_id": self.duel_id,
            "reason": self.reason,
            "run_dir": str(self.run_dir),
            "record": self.record,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> DuelResult:
        return cls(
            status=str(doc["status"]),
            kind=str(doc["kind"]),
            event_id=str(doc["event_id"]),
            duel_id=str(doc["duel_id"]),
            reason=str(doc["reason"]),
            run_dir=Path(doc["run_dir"]),
            record=doc.get("record"),
            units=list(doc.get("units") or []),
        )


@dataclass
class _Duel:
    """One run of one duel: what the phases share, and what its progress shows."""

    req: DuelRequest
    duel_id: str
    event_id: str
    runs: int
    run_dir: Path
    started_at: str
    deadline: float
    units: list[dict[str, Any]] = field(default_factory=list)
    unit_defs: list[dict[str, Any]] = field(default_factory=list)
    prepared: dict[str, PreparedSubmission] = field(default_factory=dict)
    prompts: Materialized | None = None
    sides: dict[str, dict[str, Any]] = field(default_factory=dict)
    phase: str = "fetching"
    side: str | None = None
    current: dict[str, Any] | None = None
    message: str = ""
    #: Finished clips not yet in the store: `(side, unit_id) -> path`.
    pending_media: dict[tuple[str, str], Path] = field(default_factory=dict)
    #: Why the king forfeits, when its submission was refused.
    forfeit: str | None = None
    #: How the duel ended, once it has (`progress.OUTCOMES`).
    outcome: dict[str, Any] | None = None

    def row(self, unit_id: str) -> dict[str, Any]:
        return next(u for u in self.units if u["unit_id"] == unit_id)


class Orchestrator:
    """Runs duels into one store. The caller holds the store's lock (`store_lock`)."""

    def __init__(
        self,
        spec: Spec,
        store: Store,
        runtime: Any,
        run_root: str | Path,
        *,
        progress: ProgressBoard | None = None,
        mirror: Any | None = None,
        benchmark: Any | None = None,
        workers: int = 1,
        cards: list[str | None] | None = None,
    ) -> None:
        self.spec = spec
        self.store = store
        #: Serves each side's policy (`runtime.SubprocessPolicyRuntime` or the weights runtime).
        self.runtime = runtime
        self.run_root = Path(run_root)
        #: Units run at once on each GPU, each a simulator and a policy server of its own: as many
        #: as one card's memory holds.
        self.workers = max(1, int(workers))
        #: The GPUs to use; None for the host's (`slots.gpus`), read as each duel starts, since a
        #: validator's GPU lease sets CUDA_VISIBLE_DEVICES for the step it runs a duel in.
        self.cards = list(cards) if cards is not None else None
        #: What the duel's workers share: its rows, its progress and the store.
        self._lock = threading.RLock()
        self.mirror = mirror
        #: Where the running duel's progress goes: the store's `running.json`, mirrored with it.
        self.progress = progress or ProgressBoard(store, push=self.push_touched)
        #: The RoboTwin-Vector checkout (`robotwin.RoboTwin`), found on first use; a test's fake.
        self._benchmark = benchmark

    @property
    def benchmark(self) -> Any:
        if self._benchmark is None:
            from ..robotwin import RoboTwin, Unavailable

            try:
                self._benchmark = RoboTwin.from_environment(self.spec.level)
            except Unavailable as exc:
                raise HarnessUnavailable(str(exc)) from exc
        return self._benchmark

    # -- the run directory ------------------------------------------------------------------

    def reap_orphans(self) -> list[str]:
        """End what an orchestrator killed outright left running: the process groups this run
        root's undecided duels' ledgers name. Call it holding the store's lock, before any duel
        runs; what was reaped, by name."""
        reaped = [f"process group {pgid}" for pgid in reap_run_root(self.run_root)]
        if reaped:
            log.warning("reaped what a killed orchestrator left running: %s", ", ".join(reaped))
        return reaped

    def run_dir(self, req: DuelRequest) -> Path:
        return self.run_root / req.event_id(self.spec)[:16]

    def record_request(self, req: DuelRequest) -> dict[str, Any]:
        """Write `request.json` for `req` unless it is there. A run directory started for another
        duel of the same id (the same pair and block at another runs_per_task) is moved aside as
        `<dir>.stale-<n>` and logged, rather than refused on every attempt."""
        path = self.run_dir(req) / REQUEST_FILE
        wanted = req.as_dict(self.spec)
        existing = read_json(path)
        if isinstance(existing, dict):
            if self.holds_request(req):
                return existing
            # The same pair at the same block, asked for differently: whatever that run was, it is
            # not this duel. Kept aside for its logs, never resumed as this.
            moved = move_aside(self.run_dir(req), "stale")
            log.warning("%s held another duel's request; moved it aside to %s", path, moved)
        doc = {**wanted, "started_at": now_iso()}
        atomic_write_json(path, doc)
        return doc

    def holds_request(self, req: DuelRequest) -> bool:
        """Whether `req`'s run directory was started for exactly this duel."""
        doc = read_json(self.run_dir(req) / REQUEST_FILE)
        if not isinstance(doc, dict):
            return False
        return {k: v for k, v in doc.items() if k != "started_at"} == req.as_dict(self.spec)

    def outcome(self, req: DuelRequest) -> DuelResult | None:
        doc = read_json(self.run_dir(req) / OUTCOME_FILE)
        return DuelResult.from_dict(doc) if isinstance(doc, dict) else None

    def _check_crown(self, req: DuelRequest, duel: _Duel | None = None) -> None:
        """`CrownMoved`, with the duel's run moved aside, unless the head still names the king
        `req` was asked against (no king at all, for a genesis)."""
        head = self.store.head() or {}
        holder = head.get("king") or None
        wanted = req.king.as_dict() if req.king is not None else None
        if holder == wanted:
            return
        run_dir = self.run_dir(req)
        moved = move_aside(run_dir, "stale")
        who = holder.get("repo") if isinstance(holder, dict) else "nobody"
        was = req.king.entry if req.king is not None else "nobody"
        reason = f"the crown moved from {was} to {who} since this duel was asked for"
        log.warning("duel %s is stale: %s; its run is at %s", run_dir.name, reason, moved)
        if duel is not None:
            self._post(
                duel,
                force=True,
                phase="failed",
                side=None,
                current=None,
                message=f"stale: {reason}",
                outcome={"status": "stale", "reason": reason},
            )
        raise CrownMoved(req, reason, moved)

    # -- the duel ---------------------------------------------------------------------------

    def run(self, req: DuelRequest) -> DuelResult:
        """Run `req` to a published record or a void outcome, resuming what an earlier run left.

        `DuelFailed` when the harness cannot; an interrupt (`KeyboardInterrupt`, a kill) passes
        through untouched, and running the duel again resumes it.
        """
        request = self.record_request(req)
        decided = self.outcome(req)
        if decided is not None:
            log.info("duel %s was already decided: %s", decided.event_id[:16], decided.status)
            return decided
        existing = self._published(req.event_id(self.spec))
        if existing is not None:
            return self._republished(req, existing)
        self._check_crown(req)
        run_dir = self.run_dir(req)
        # The duel's budget is spent by work, not by the time a stopped orchestrator was down.
        spent = _work_seconds(run_dir)
        duel = _Duel(
            req=req,
            duel_id=req.duel_id(self.spec),
            event_id=req.event_id(self.spec),
            runs=req.runs(self.spec),
            run_dir=run_dir,
            started_at=str(request["started_at"]),
            deadline=time.monotonic() + self.spec.budgets["duel"] - spent,
        )
        try:
            result = self._run(duel)
        except CrownMoved:
            raise
        except DuelFailed as exc:
            self._failed(duel, str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - reported, then raised as the harness's failure
            log.exception("duel %s crashed", duel.event_id[:16])
            self._failed(duel, f"{type(exc).__name__}: {exc}")
            raise DuelFailed(f"{type(exc).__name__}: {exc}") from exc
        (run_dir / FAILED_FILE).unlink(missing_ok=True)
        return result

    def _run(self, duel: _Duel) -> DuelResult:
        spec, req = self.spec, duel.req
        self._derive(duel)
        sides = self._sides(req)

        # ---- fetching and checking: a refused challenger decides the duel before anything runs;
        # a refused king forfeits, once, and a resumed duel keeps the forfeit it recorded
        refused: dict[str, str] = {}
        forfeit = read_json(duel.run_dir / FORFEIT_FILE)
        if req.king is not None and isinstance(forfeit, dict) and forfeit.get("reason"):
            refused["king"] = str(forfeit["reason"])
        self._post(duel, force=True, phase="fetching", message="fetching the submissions")
        fetched: dict[str, Any] = {}
        for side, ref in sides:
            if side in refused:
                continue
            try:
                fetched[side] = self.runtime.fetch(ref, workdir=duel.run_dir / side)
            except SubmissionRefused as exc:
                refused[side] = str(exc)
                if side == "challenger":
                    break
            except RuntimeUnavailable as exc:
                raise HarnessUnavailable(f"fetching the {side}: {exc}") from exc
        self._post(duel, force=True, phase="checking", message="checking and building")
        for side, _ in sides:
            if "challenger" in refused:
                break
            if side in refused or side not in fetched:
                continue
            self._post(duel, message=f"checking the {side}")
            try:
                duel.prepared[side] = self.runtime.prepare(
                    fetched[side], workdir=duel.run_dir / side / "check"
                )
            except SubmissionRefused as exc:
                refused[side] = str(exc)
            except RuntimeUnavailable as exc:
                raise HarnessUnavailable(f"checking the {side}: {exc}") from exc
        for side, ref in sides:
            duel.sides[side] = self._side_meta(duel, side, ref, refused.get(side))
        if "challenger" in refused:
            return self._refused(
                duel, f"the challenger's submission was refused: {refused['challenger']}"
            )
        duel.forfeit = refused.get("king")
        if duel.forfeit is not None:
            log.warning("duel %s: the king forfeits: %s", duel.event_id[:16], duel.forfeit)
            atomic_write_json(duel.run_dir / FORFEIT_FILE, {"side": "king", "reason": duel.forfeit})

        # ---- materializing and evaluating, together in the slots
        self._post(
            duel, force=True, phase="materializing", message="materializing the demonstrations"
        )
        duel.prompts = Materialized(root=duel.run_dir / "prompts", prompts={}, done=False)
        cards = self.cards if self.cards is not None else gpus()
        slots = Slots(self.workers, cards, duel.run_dir / "slots")
        log.info("duel %s runs on %d slot(s): %s", duel.event_id[:16], len(slots), cards)
        abort = threading.Event()
        try:
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="duel") as phases:
                jobs = [phases.submit(self._materialize, duel, slots, abort)]
                jobs += [
                    phases.submit(self._evaluate, duel, side, slots, abort) for side, _ in sides
                ]
                try:
                    wait(jobs, return_when=FIRST_EXCEPTION)
                    for job in jobs:
                        if job.done() and job.exception() is not None:
                            job.result()
                    for job in jobs:
                        job.result()
                except BaseException:
                    # An interrupt, or one phase failing: end what the slots run - none of it is
                    # recorded - so the other phases stop too, and the duel resumes from here.
                    abort.set()
                    duel.prompts.finish(failed=True)
                    RUNNING.stop()
                    slots.close()
                    raise
        finally:
            slots.close()
            RUNNING.resume()
        self._flush_media(duel, force=True)
        self.push_touched()
        if score.too_void(duel.units, spec.max_void):
            void = sum(1 for u in duel.units if u["void"])
            prompts = duel.prompts.void
            what = (
                f"{prompts} of {len(duel.units)} prompts are void"
                if prompts and prompts == void
                else f"{void} of {len(duel.units)} units are void"
            )
            return self._void(duel, what)
        return self._publish(duel)

    # -- phases -----------------------------------------------------------------------------

    def _derive(self, duel: _Duel) -> None:
        """The units: a checkout that is missing or cannot run the spec's tasks stops the duel
        before anything is fetched, and the duel waits for it."""
        from ..robotwin import Unavailable

        try:
            self.benchmark.prepare(self.spec.tasks)
        except Unavailable as exc:
            raise HarnessUnavailable(str(exc)) from exc
        duel.unit_defs = deal(self.spec, duel.duel_id, duel.req.entropy, duel.runs)
        duel.units = [unit_row(unit) for unit in duel.unit_defs]

    def _sides(self, req: DuelRequest) -> list[tuple[str, SubmissionRef]]:
        sides = [("challenger", req.challenger)]
        if req.king is not None:
            sides.append(("king", req.king))
        return sides

    def _materialize(self, duel: _Duel, slots: Slots, abort: threading.Event) -> None:
        materialize_units(
            self.spec,
            duel.unit_defs,
            duel.run_dir / "prompts",
            benchmark=self.benchmark,
            deadline=duel.deadline,
            stop=lambda: self._certainly_void(duel),
            on_prompt=lambda unit, prompt: self._on_prompt(duel, unit, prompt),
            stall_s=self.spec.budgets["stall"],
            stall_retries=self.spec.stall_retries,
            slots=slots,
            into=duel.prompts,
            abort=abort,
        )

    def _on_prompt(self, duel: _Duel, unit: Mapping[str, Any], prompt: Prompt) -> None:
        with self._lock:
            self._record_prompt(duel, prompt)

    def _record_prompt(self, duel: _Duel, prompt: Prompt) -> None:
        row = duel.row(prompt.unit_id)
        if prompt.void:
            row["void"] = True
            row["king_error"] = row["challenger_error"] = prompt.error
        else:
            row["prompt_sha256"] = prompt.sha256
            # The scene both sides played: the benchmark chooses it only while materializing,
            # among the candidates the unit names.
            row["scene_seed"] = prompt.scene_seed
            if prompt.demo_clip is not None:
                row["demo_video"] = self.store.put_media(prompt.demo_clip)
        done = sum(1 for u in duel.units if u["prompt_sha256"] or u["void"])
        self._post(duel, message=f"materialized {done} of {len(duel.units)}: {prompt.unit_id}")

    def _certainly_void(self, duel: _Duel) -> bool:
        """Whether the duel is void whatever its remaining units do: void units only ever add up,
        so once more than `crown.max_void` of all units are void, nothing played or materialized
        after can make it stand."""
        with self._lock:
            return score.too_void(duel.units, self.spec.max_void)

    def _evaluate(self, duel: _Duel, side: str, slots: Slots, abort: threading.Event) -> None:
        side_dir = duel.run_dir / side
        with self._lock:
            # What an earlier run of this duel finished is merged before anything new runs.
            for record in read_results(side_dir).values():
                self._merge(duel, side, record)
        run_side(
            self.spec,
            side=side,
            units=duel.unit_defs,
            prompts=duel.prompts or Materialized(root=duel.run_dir / "prompts", prompts={}),
            side_dir=side_dir,
            benchmark=self.benchmark,
            runtime=self.runtime,
            prepared=duel.prepared.get(side),
            refused=duel.forfeit if side == "king" else None,
            deadline=duel.deadline,
            void_units=_VoidOn(self, duel, _other(side)),
            stop=lambda: self._certainly_void(duel),
            on_start=lambda unit: self._on_start(duel, side, unit),
            on_unit=lambda unit, record: self._on_unit(duel, side, record),
            stall_s=self.spec.budgets["stall"],
            stall_retries=self.spec.stall_retries,
            slots=slots,
            abort=abort,
        )

    def _on_start(self, duel: _Duel, side: str, unit: Mapping[str, Any]) -> None:
        with self._lock:
            duel.current = {"unit_id": unit["unit_id"], "side": side}
            self._post(duel, phase="evaluating", side=side, message=f"{side}: {unit['unit_id']}")

    def _on_unit(self, duel: _Duel, side: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._merge(duel, side, record)
            self._flush_media(duel)
            state = "void" if record["void"] else ("ok" if record["success"] else "fail")
            duel.current = None
            self._post(duel, message=f"{side}: {record['unit_id']} {state}")

    def _merge(self, duel: _Duel, side: str, record: Mapping[str, Any]) -> None:
        """One side's unit record into the published row. Void on either side is void for both."""
        row = duel.row(str(record["unit_id"]))
        row[f"{side}_error"] = record.get("error")
        if record.get("void"):
            row["void"] = True
            row[f"{side}_success"] = None
        else:
            row[f"{side}_success"] = bool(record.get("success"))
            row[f"{side}_steps"] = record.get("steps")
            clip = record.get("clip")
            if clip and not row.get(f"{side}_video"):
                duel.pending_media[(side, row["unit_id"])] = duel.run_dir / side / str(clip)
        if row["void"]:
            # Neither side is scored on it, and the record does not show one as if it were; what
            # the other side did stays in its results file and its error.
            row["king_success"] = row["challenger_success"] = None
            row["outcome"] = "tie"
        else:
            row["outcome"] = score.paired_outcome(
                row.get("king_success"), row.get("challenger_success")
            )

    def _flush_media(self, duel: _Duel, *, force: bool = False) -> None:
        """Finished clips into the store every `MEDIA_FLUSH_UNITS` units, so the dashboard can
        play them while the duel runs; mirrored when there is a mirror."""
        with self._lock:
            self._flush_media_locked(duel, force)

    def _flush_media_locked(self, duel: _Duel, force: bool) -> None:
        if not duel.pending_media or (not force and len(duel.pending_media) < MEDIA_FLUSH_UNITS):
            return
        for (side, unit_id), path in list(duel.pending_media.items()):
            del duel.pending_media[(side, unit_id)]
            if path.is_file():
                duel.row(unit_id)[f"{side}_video"] = self.store.put_media(path)
        self.push_touched()

    def _side_meta(
        self, duel: _Duel, side: str, ref: SubmissionRef, refused: str | None
    ) -> dict[str, Any]:
        """A side as its event publishes it: the submission, the sha256 of the weights served,
        and why it was refused when it was."""
        prepared = duel.prepared.get(side)
        return {
            **ref.as_dict(),
            "weights_sha256": prepared.weights_sha256 if prepared else None,
            "refused": refused,
        }

    def _publish(self, duel: _Duel) -> DuelResult:
        spec, req = self.spec, duel.req
        self._post(
            duel,
            force=True,
            phase="publishing",
            side=None,
            current=None,
            message="publishing the record",
        )
        existing = self._published(duel.event_id)
        if existing is not None:
            return self._republished(req, existing)
        self._check_crown(req, duel)
        if req.kind == "genesis":
            # The entrant ran as the challenger and is published as the king it becomes.
            units = [_as_king(u) for u in duel.units]
            king_scores, challenger_scores = score.task_scores(units, "king", spec.tasks), None
            dethroned, reason = False, "genesis"
            king, challenger = req.challenger, None
            sides = {"king": duel.sides.get("challenger", {}), "challenger": None}
            verdict: dict[str, Any] = {"reason": reason}
        else:
            units = duel.units
            result = score.verdict(units, spec.margin, spec.tasks)
            king_scores, challenger_scores = result.king_scores, result.challenger_scores
            dethroned, reason = result.dethroned, result.reason
            king, challenger = req.king, req.challenger
            sides = {
                "king": duel.sides.get("king", {}),
                "challenger": duel.sides.get("challenger", {}),
            }
            verdict = result.as_dict()
        finished_at = now_iso()
        record = index_record(
            event_id=duel.event_id,
            kind=req.kind,
            block=req.block,
            finished_at=finished_at,
            king=king.as_dict(),
            challenger=challenger.as_dict() if challenger else None,
            king_score=king_scores["average"],
            challenger_score=challenger_scores["average"] if challenger_scores else None,
            dethroned=dethroned,
            units=units,
        )
        event = {
            "event_id": duel.event_id,
            "duel_id": duel.duel_id,
            "kind": req.kind,
            "block": req.block,
            # The chain block whose hash seeded the units: anyone can read it back and deal again.
            "seed": {"block": int(req.seed_block), "hash": req.seed_block_hash}
            if req.seed_block is not None
            else None,
            "spec_version": spec.version,
            "runs_per_task": duel.runs,
            "started_at": duel.started_at,
            "finished_at": finished_at,
            "king": sides["king"],
            "challenger": sides["challenger"],
            "scores": {"king": king_scores, "challenger": challenger_scores},
            "dethroned": dethroned,
            "verdict": {**verdict, "void_fraction": score.void_fraction(units)},
            "benchmark": self._benchmark_info(),
            "notes": [f"king forfeit: {duel.forfeit}"] if duel.forfeit else [],
            "units": units,
        }
        self.store.write_event(event)
        record["seq"] = self.store.append(record)
        self.push_touched()
        result_ = DuelResult(
            status="published",
            kind=req.kind,
            event_id=duel.event_id,
            duel_id=duel.duel_id,
            reason=reason,
            run_dir=duel.run_dir,
            record=record,
            units=units,
        )
        atomic_write_json(duel.run_dir / OUTCOME_FILE, result_.as_dict())
        moved = "moves" if dethroned or req.kind == "genesis" else "stays"
        self._post(
            duel,
            force=True,
            phase="done",
            message=f"the crown {moved}: {reason}",
            outcome={
                "status": "published",
                "dethroned": dethroned,
                "reason": reason,
                "king_score": king_scores["average"],
                "challenger_score": challenger_scores["average"] if challenger_scores else None,
            },
        )
        return result_

    def _republished(self, req: DuelRequest, record: dict[str, Any]) -> DuelResult:
        """A duel an earlier run published before it stopped: nothing is fetched, run or published
        again. Its head is rebuilt from the index - a kill between the index line and the head's
        rewrite leaves the head naming the king before it, whom the next duel would face - and its
        files are pushed to the mirror again, since a kill before the push leaves them off it."""
        eid = str(record["event_id"])
        log.info(
            "duel %s is published as seq %s already; healing its head", eid[:16], record["seq"]
        )
        self.store.rebuild_head()
        event = self.store.event(eid) or {}
        units = [u for u in event.get("units") or [] if isinstance(u, dict)]
        paths = [
            self.store.event_path(eid),
            self.store.index_part_path(self.store.part_of(int(record["seq"]))),
            self.store.head_path(),
            *(self.store.media_path(sha) for sha in media_shas(units)),
        ]
        for path in paths:
            if path.is_file():
                self.store.touch(path)
        self.push_touched()
        verdict = event.get("verdict") if isinstance(event.get("verdict"), dict) else {}
        result = DuelResult(
            status="published",
            kind=str(record.get("kind") or req.kind),
            event_id=eid,
            duel_id=req.duel_id(self.spec),
            reason=str(verdict.get("reason") or record.get("kind")),
            run_dir=self.run_dir(req),
            record=record,
            units=units,
        )
        atomic_write_json(self.run_dir(req) / OUTCOME_FILE, result.as_dict())
        return result

    def _refused(self, duel: _Duel, reason: str) -> DuelResult:
        """A refused challenger: nothing runs, nothing is published, and the king is not asked."""
        log.warning("duel %s is refused: %s", duel.event_id[:16], reason)
        for row in duel.units:
            row["challenger_success"], row["challenger_error"] = False, reason
        return self._decided(duel, "refused", reason)

    def _void(self, duel: _Duel, reason: str) -> DuelResult:
        log.warning("duel %s is void: %s", duel.event_id[:16], reason)
        return self._decided(duel, "void", reason)

    def _decided(self, duel: _Duel, status: str, reason: str) -> DuelResult:
        result = DuelResult(
            status=status,
            kind=duel.req.kind,
            event_id=duel.event_id,
            duel_id=duel.duel_id,
            reason=reason,
            run_dir=duel.run_dir,
            units=duel.units,
        )
        atomic_write_json(duel.run_dir / OUTCOME_FILE, {**result.as_dict(), "sides": duel.sides})
        self._post(
            duel,
            force=True,
            phase="failed",
            side=None,
            current=None,
            message=f"{status}: {reason}",
            outcome={"status": status, "reason": reason},
        )
        return result

    def _failed(self, duel: _Duel, reason: str) -> None:
        log.error("duel %s failed: %s", duel.event_id[:16], reason)
        try:
            (duel.run_dir / FAILED_FILE).write_text(reason + "\n", encoding="utf-8")
        except OSError:
            pass
        self._post(
            duel,
            force=True,
            phase="failed",
            side=None,
            current=None,
            message=f"failed: {reason}",
            outcome={"status": "failed", "reason": reason},
        )

    # -- plumbing ---------------------------------------------------------------------------

    def _published(self, eid: str) -> dict[str, Any] | None:
        """The index's record of this event, if an earlier run published it before it stopped."""
        for record in reversed(self.store.iter_index()):
            if record.get("event_id") == eid:
                return record
        return None

    def _benchmark_info(self) -> dict[str, Any]:
        try:
            return dict(self.benchmark.info())
        except Exception as exc:  # noqa: BLE001 - recorded, since the duel already ran
            return {"error": f"{type(exc).__name__}: {exc}"}

    def push_touched(self) -> None:
        """What the store wrote since the last push, to the mirror when there is one."""
        if self.mirror is None:
            return
        files = self.store.drain_touched()
        if not files:
            return
        try:
            self.mirror.push(files)
        except Exception as exc:  # noqa: BLE001 - mirroring is retried with the next push
            log.warning("mirror push failed: %s", exc)
            self.store.touched.update(files)

    def _post(self, duel: _Duel, *, force: bool = False, **changes: Any) -> None:
        with self._lock:
            self._post_locked(duel, force, changes)

    def _post_locked(self, duel: _Duel, force: bool, changes: Mapping[str, Any]) -> None:
        for key, value in changes.items():
            setattr(duel, key, value)
        req = duel.req
        try:
            doc = build_progress(
                event_id=duel.event_id,
                kind=req.kind,
                king=req.king.as_dict() if req.king else None,
                challenger=req.challenger.as_dict(),
                phase=duel.phase,
                side=duel.side,
                units=duel.units,
                current=duel.current,
                message=duel.message,
                started_at=duel.started_at,
                outcome=duel.outcome,
            )
        except ValueError:
            log.exception("the duel's progress could not be built")
            return
        self.progress.post(doc, force=force)


def _other(side: str) -> str:
    return "king" if side == "challenger" else "challenger"


def _as_king(row: Mapping[str, Any]) -> dict[str, Any]:
    """A genesis row: the entrant ran as the challenger and is published as the king it becomes."""
    out = dict(row)
    for field_ in ("success", "video", "steps", "error"):
        out[f"king_{field_}"] = row.get(f"challenger_{field_}")
        out[f"challenger_{field_}"] = None
    out["outcome"] = "tie"
    return out


class _VoidOn(Mapping[str, str]):
    """The units a side's results made void, read live from the duel's rows: the other side does
    not play them."""

    def __init__(self, orchestrator: Orchestrator, duel: _Duel, side: str) -> None:
        self._orchestrator, self._duel, self._side = orchestrator, duel, side

    def _void(self) -> dict[str, str]:
        with self._orchestrator._lock:
            return {
                str(
                    row["unit_id"]
                ): f"void on the {self._side}'s side: {row.get(f'{self._side}_error')}"
                for row in self._duel.units
                if row["void"] and row.get(f"{self._side}_error") is not None
            }

    def __getitem__(self, unit_id: str) -> str:
        return self._void()[unit_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._void())

    def __len__(self) -> int:
        return len(self._void())


def _work_seconds(run_dir: Path) -> float:
    """The wall time an earlier run of this duel spent materializing and playing units."""
    from .materialize import read_manifest

    spent = sum(p.wall_s for p in read_manifest(run_dir / "prompts").values())
    for side in SIDES:
        spent += sum(float(r.get("wall_s") or 0.0) for r in read_results(run_dir / side).values())
    return spent


def _seconds_since(iso: str) -> float:
    try:
        started = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
