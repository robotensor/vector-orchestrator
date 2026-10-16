"""`vector-orchestrator`: run a duel, prepare a store, check the benchmark.

    vector-orchestrator check
    vector-orchestrator store init   --root DIR
    vector-orchestrator store mirror --root DIR --repo OWNER/NAME [--prune]
    vector-orchestrator duel --store DIR --run-dir DIR --challenger OWNER/NAME@REVISION [...]

Every import is local to its command, so `--help` works on a host with nothing else installed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _spec(args: argparse.Namespace):
    from .spec import load_spec

    return load_spec(args.spec)


def cmd_check(args: argparse.Namespace) -> int:
    """Whether this host's RoboTwin-Vector checkout can run the spec's tasks."""
    from .robotwin import RoboTwin, Unavailable

    spec = _spec(args)
    try:
        fork = RoboTwin.from_environment(spec.level)
        fork.prepare(spec.tasks)
    except Unavailable as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"ok": True, "root": fork.workdir, **fork.info()}, indent=2))
    return 0


def cmd_store(args: argparse.Namespace) -> int:
    spec = _spec(args)
    if args.store_cmd == "init":
        from .store.writer import Store, store_lock

        root = Path(args.root).resolve()
        with store_lock(root):
            manifest = Store(root, spec).init()
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    from .store.mirror import MirrorUnreachable, PushRefused, mirror_store

    try:
        n = mirror_store(args.root, args.repo, message=args.message, prune=args.prune)
    except (ValueError, PushRefused, MirrorUnreachable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"mirrored {n} files to {args.repo}")
    return 0


def _orchestrator(args: argparse.Namespace, spec):
    """The store, runtime and mirror a duel publishes with."""
    from .duel.orchestrate import Orchestrator
    from .store.writer import Store

    store = Store(args.store, spec)
    if store.manifest() is None:
        raise ValueError(f"{args.store} is not a store; run `vector-orchestrator store init` first")
    if args.local:
        from .duel.runtime import SubprocessPolicyRuntime

        runtime = SubprocessPolicyRuntime(
            spec, _pairs(args.local, "--local"), python=args.policy_python
        )
    else:
        from .duel.weights_runtime import WeightsPolicyRuntime

        runtime = WeightsPolicyRuntime(
            spec,
            python=args.policy_python,
            cache_root=args.cache,
            kwargs=_pairs(args.policy_kwarg, "--policy-kwarg"),
        )
    mirror = None
    if args.mirror:
        from .store.mirror import Mirror

        mirror = Mirror(store.root, args.mirror, token=os.environ.get("HF_TOKEN"))
    return Orchestrator(
        spec, store, runtime, args.run_dir, mirror=mirror, workers=args.workers,
        free_gb=args.free_gb,
    )  # fmt: skip


def _pairs(pairs: list[str] | None, flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key or key == "weights":
            raise ValueError(f"{flag} {pair!r} is not KEY=VALUE")
        out[key] = value
    return out


def _logging() -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


class Terminated(KeyboardInterrupt):
    """SIGTERM or SIGINT, raised where the duel was when it arrived."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(signal.Signals(self.signum).name)


@contextmanager
def _terminable() -> Iterator[None]:
    """SIGTERM and SIGINT raise `Terminated` in the main thread, so what was running unwinds
    through its `finally` blocks - the unit's policy server, the benchmark's process group, the
    socket directory - before the process exits. Python's default for SIGTERM is to
    die on the spot, which leaves all of those running. A second signal while that unwinding
    happens is ignored; SIGKILL still ends the process, and the next start reaps what it left."""

    def stop(signum: int, frame: Any) -> None:
        for name in (signal.SIGTERM, signal.SIGINT):
            signal.signal(name, signal.SIG_IGN)
        raise Terminated(signum)

    previous = {name: signal.signal(name, stop) for name in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def _until_terminated(
    command: Callable[[argparse.Namespace], int], args: argparse.Namespace
) -> int:
    try:
        with _terminable():
            return command(args)
    except Terminated as exc:
        print(
            f"stopped by {exc}: the unit that was running was torn down; the same command resumes "
            "the duel where it stopped",
            file=sys.stderr,
        )
        return 128 + exc.signum


def cmd_duel(args: argparse.Namespace) -> int:
    try:
        with _terminable():
            return _duel(args)
    except Terminated as exc:
        print(
            f"stopped by {exc}: the same command resumes the duel where it stopped", file=sys.stderr
        )
        return 128 + exc.signum


def _duel(args: argparse.Namespace) -> int:
    from .duel.orchestrate import DuelFailed, DuelRequest
    from .duel.runtime import RuntimeUnavailable, SubmissionRefused
    from .ids import SubmissionRef
    from .store.writer import store_lock

    spec = _spec(args)
    try:
        repo, _, revision = args.challenger.partition("@")
        if not repo or not revision:
            raise ValueError(f"{args.challenger!r} is not owner/name@revision")
        orchestrator = _orchestrator(args, spec)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _logging()
    try:
        challenger = orchestrator.runtime.resolve(repo, revision)
    except (RuntimeUnavailable, SubmissionRefused, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    store = orchestrator.store
    try:
        with store_lock(store.root):
            orchestrator.reap_orphans()
            head = store.head() or {}
            king = SubmissionRef.from_dict(head.get("king"))
            if king is not None and king.key == challenger.key:
                print(f"error: {challenger.entry} already holds the crown", file=sys.stderr)
                return 2
            # The next block unless told: running a duel that did not publish again resumes it.
            block = args.block if args.block is not None else int(head.get("block") or 0) + 1
            req = DuelRequest(
                challenger,
                king,
                block,
                seed_block=args.seed_block,
                seed_block_hash=args.seed_block_hash,
                runs_per_task=args.runs_per_task,
            )
            result = orchestrator.run(req)
    except (DuelFailed, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    record = result.record or {}
    summary = {
        "status": result.status,
        "kind": result.kind,
        "reason": result.reason,
        "event_id": result.event_id,
        "dethroned": record.get("dethroned"),
        "king_score": record.get("king_score"),
        "challenger_score": record.get("challenger_score"),
        "void": sum(1 for u in result.units if u.get("void")),
        "units": len(result.units),
        "run_dir": str(result.run_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.published else 1


def cmd_reap(args: argparse.Namespace) -> int:
    """One pass of what the orchestrator does beside its duels: finished duels under the run dir
    cleared, oldest first, until the free space asked for is there."""
    from .duel import storage

    _logging()
    free_gb = storage.free_gb_from_env() if args.free_gb is None else max(0.0, args.free_gb)
    before = storage.free_bytes(args.run_dir)
    cleared = storage.reap(args.run_dir, free_gb, dry_run=args.dry_run)
    verb = "would clear" if args.dry_run else "cleared"
    for duel, held in cleared:
        print(f"{verb} {duel.path.name} ({duel.how}): {held / storage.GB:.2f} GB")
    after = (
        before + sum(h for _, h in cleared) if args.dry_run else storage.free_bytes(args.run_dir)
    )
    print(
        f"{before / storage.GB:.1f} GB free before, {after / storage.GB:.1f} GB after; "
        f"keeping {free_gb:g} GB free"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vector-orchestrator", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--spec", default=None, help="the spec (default: $VECTOR_ORCHESTRATOR_SPEC, else spec.json)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ck = sub.add_parser(
        "check", help="can this host's RoboTwin-Vector checkout run the spec's tasks"
    )
    ck.set_defaults(func=cmd_check)

    st = sub.add_parser("store", help="the result store")
    st_sub = st.add_subparsers(dest="store_cmd", required=True)
    st_init = st_sub.add_parser("init", help="write the manifest and an empty head")
    st_init.add_argument("--root", required=True)
    st_mirror = st_sub.add_parser("mirror", help="push the store to a Hugging Face dataset")
    st_mirror.add_argument("--root", required=True)
    st_mirror.add_argument("--repo", required=True)
    st_mirror.add_argument("--message", default=None)
    st_mirror.add_argument(
        "--prune", action="store_true", help="delete files the store no longer has"
    )
    st.set_defaults(func=cmd_store)

    du = sub.add_parser("duel", help="run one duel (a genesis with no king) and publish it")
    du.add_argument("--store", required=True, help="the store to publish to (`store init`)")
    du.add_argument(
        "--run-dir", required=True, help="where duels keep prompts and logs, and resume from"
    )
    du.add_argument("--challenger", required=True, metavar="OWNER/NAME@REVISION")
    du.add_argument(
        "--block", type=int, default=None, help="the duel's block (default: the head's + 1)"
    )
    du.add_argument(
        "--seed-block", type=int, default=None, help="the chain block whose hash seeds the units"
    )
    du.add_argument("--seed-block-hash", default=None, help="that block's hash, 0x + 64 hex")
    du.add_argument(
        "--runs-per-task", type=int, default=None, help="units per task (default: the spec's)"
    )
    du.add_argument(
        "--workers",
        type=int,
        default=1,
        help="units run at once on each GPU (CUDA_VISIBLE_DEVICES, else every card nvidia-smi "
        "lists), each a simulator and a policy server: as many as one card's memory holds "
        "(default 1)",
    )
    du.add_argument(
        "--policy-python", default=os.environ.get("VECTOR_POLICY_PYTHON") or sys.executable
    )
    du.add_argument(
        "--policy-kwarg", action="append", metavar="KEY=VALUE", help="e.g. device=cuda:0"
    )
    du.add_argument("--cache", default="cache", help="where downloaded weights are kept, by commit")
    du.add_argument(
        "--local", action="append", metavar="REPO=DIR", help="serve REPO's policy.yaml from DIR"
    )
    du.add_argument(
        "--mirror", default=None, metavar="REPO", help="push what is published to this dataset"
    )
    du.add_argument(
        "--free-gb",
        type=float,
        default=None,
        help="GB to keep free on the run dir's filesystem by clearing finished duels, oldest "
        "first (default: $VECTOR_FREE_GB, else 100; 0 for never)",
    )
    du.set_defaults(func=cmd_duel)

    rp = sub.add_parser(
        "reap", help="clear what finished duels left under the run dir, oldest first, for room"
    )
    rp.add_argument("--run-dir", required=True, help="where duels keep prompts and logs")
    rp.add_argument(
        "--free-gb",
        type=float,
        default=None,
        help="GB to have free when done (default: $VECTOR_FREE_GB, else 100)",
    )
    rp.add_argument(
        "--dry-run", action="store_true", help="say what would be cleared, and clear nothing"
    )
    rp.set_defaults(func=cmd_reap)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
