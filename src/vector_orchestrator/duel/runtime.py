"""Serving a side's policy: `python -m vector_protocol.serve` on this host.

A duel asks a runtime, in order: `resolve(repo, revision)` for the commit that will run; `fetch`
then `prepare` once per side (the `fetching` and `checking` phases); and `serve(prepared)`, a
context manager serving the policy for one unit. Alone, each `serve` is a fresh server that
accepts one client and exits. Given a `ServerPool`, a server is kept between units
(`vector_protocol.serve --keep`): the next unit of the same submission is served by the same
process, with its model already built, and `reset` and `prompt` start its episode; a unit of
another submission, or a server that died, gets a fresh one. `env` puts the server on a card
(`CUDA_VISIBLE_DEVICES`).

Whose a failure is decides what it costs. A submission that cannot run raises `SubmissionRefused`
(a published reason); a policy that never listens within `budgets.policy_start` raises
`PolicyDied` (that side's failure on the unit); a runtime that cannot do its part raises
`RuntimeUnavailable` (nobody's loss, void for both). After a unit, `ServedPolicy.died()` says how
the policy ended if it did not end cleanly.

`SubprocessPolicyRuntime` serves local directories holding a `policy.yaml` (the example policies,
the tests); `weights_runtime.WeightsPolicyRuntime` serves submissions.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vector_protocol.client import RemotePolicy
from vector_protocol.errors import ManifestError, PolicyUnavailable
from vector_protocol.logs import tail
from vector_protocol.manifest import load as load_manifest

from ..ids import SubmissionRef, is_commit_sha
from ..runner import benchmark_environment
from .orphans import Ledger


class SubmissionRefused(Exception):
    """The submission cannot run, for a reason of its own; `step` says where it stopped."""

    def __init__(self, step: str, reason: str) -> None:
        self.step = step
        self.reason = reason
        super().__init__(f"{step}: {reason}")


class RuntimeUnavailable(RuntimeError):
    """The runtime could not take a submission through a step (no Hub, no server); nobody's loss."""


class PolicyDied(RuntimeError):
    """A served policy never listened: it exited first, or did not listen in time. Its side's
    failure on the unit, never a void."""


#: Who brought about a unit's end: the side's own submission, or the harness around it. The same
#: two words a benchmark's `read_result` may give as `void_cause`.
POLICY = "policy"
HARNESS = "harness"
CAUSES = (POLICY, HARNESS)

#: How much of a policy's log the copy a benchmark reads holds: its end, which is all a failure
#: quotes (`vector_protocol.logs.tail` reads at most 8 KiB).
LOG_TAIL_BYTES = 1 << 16
#: How often that copy catches up with the log.
LOG_MIRROR_S = 0.05

#: A policy can write where its log is, so it may have made the log a link or a pipe: the log is
#: opened without following a link or waiting on a pipe, and read only if it is a regular file.
_LOG_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


@dataclass(frozen=True)
class PolicyEnd:
    """How a unit's policy ended underneath it, when it did not end cleanly.

    `cause` is `policy` for an end its submission can bring about - a non-zero exit, a signal,
    running out of memory - and `harness` for one it cannot.
    """

    reason: str
    cause: str


@dataclass(frozen=True)
class FetchedSubmission:
    """A submission at the commit it resolved to, where the runtime can serve it from."""

    ref: SubmissionRef
    commit: str
    root: Path


@dataclass(frozen=True)
class PreparedSubmission:
    """A submission ready to serve: checked, and answering `hello`."""

    ref: SubmissionRef
    commit: str
    #: The weights file's sha256, published with the side; None for a local directory.
    weights_sha256: str | None
    #: What the runtime serves it from (its manifest); not published.
    handle: Any = None


#: The variable a served policy's key travels in, by name only.
AUTHKEY_ENV = "VECTOR_AUTHKEY"
AUTHKEY_BYTES = 32
SOCKET_FILE = "policy.sock"
#: In the unit's directory once the unit is over: the server's log and what the policy printed.
LOG_FILE = "policy.log"
#: The most of a policy's log copied out: the policy writes it and nothing else bounds it.
MAX_LOG_BYTES = 16 << 20
POLL_S = 0.05
#: How long a server whose client has gone gets to exit on its own before its status is read.
EXIT_GRACE_S = 5.0
#: What a local directory holds that is not the submission.
IGNORED = frozenset({".git", "__pycache__"})
#: The manifest a served directory holds, and its schema.
MANIFEST = "policy.yaml"
MANIFEST_API = 1


def local_directories(pairs: Mapping[str, str | os.PathLike[str]]) -> dict[str, Path]:
    return {str(k): Path(v).resolve() for k, v in pairs.items()}


def tree_hash(directory: Path) -> str:
    """A 40-hex address of a directory's files and bytes, standing in for a commit sha."""
    digest = hashlib.sha1()
    paths = []
    for root, dirs, files in os.walk(directory):  # never follows a link to a directory
        dirs[:] = [d for d in dirs if d not in IGNORED]
        paths += [Path(root) / name for name in (*files, *dirs)]
    for path in sorted(paths):
        rel = path.relative_to(directory)
        info = os.lstat(path)
        if stat.S_ISREG(info.st_mode):
            content = hashlib.sha1(path.read_bytes()).hexdigest()
        elif stat.S_ISLNK(info.st_mode):
            content = "link:" + os.readlink(path)
        else:
            continue
        digest.update(f"{rel.as_posix()}\0{content}\n".encode())
    return digest.hexdigest()


def copy_log(source: Path, target: Path, limit: int, start: int = 0) -> None:
    """Append at most `limit` bytes of a policy's log, from byte `start`, to `target`. The policy
    can write where its log is, so it may have made it a link or a pipe: only a regular file is
    read, and nothing is followed or waited on."""
    try:
        fd = os.open(source, _LOG_FLAGS)
    except OSError:
        return
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return
        with open(fd, "rb", closefd=False) as fh, open(target, "ab") as out:
            fh.seek(start)
            out.write(fh.read(limit))
    except OSError:
        pass
    finally:
        os.close(fd)


def _copy_tail(source: Path, target: Path, limit: int, seen: Any, start: int = 0) -> Any:
    """Replace `target` with the last `limit` bytes of the log at `source` from byte `start` on,
    unless the log is as it was when `seen` was returned; what to pass as `seen` next time. A log
    that is not a regular file, or cannot be read, leaves `target` as it was."""
    try:
        fd = os.open(source, _LOG_FLAGS)
    except OSError:
        return seen
    try:
        info = os.fstat(fd)
        state = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if not stat.S_ISREG(info.st_mode) or state == seen:
            return seen
        with open(fd, "rb", closefd=False) as fh:
            fh.seek(max(start, info.st_size - limit))
            data = fh.read(limit)
        partial = target.with_name(f".{target.name}.partial")
        partial.write_bytes(data)
        os.replace(partial, target)  # a reader never sees half of a copy
        return state
    except OSError:
        return seen
    finally:
        os.close(fd)


@contextmanager
def mirror_log(
    source: Path,
    target: Path,
    *,
    limit: int = LOG_TAIL_BYTES,
    interval_s: float = LOG_MIRROR_S,
    start: int = 0,
) -> Iterator[Path]:
    """`target`, holding the end of the policy's log at `source` for as long as the block runs.

    A benchmark quotes a policy's log when the policy fails it, and must not be handed the log
    itself: the policy can write where its log is, so it could swap the file for a link to
    something on the benchmark's side, or a pipe that blocks whoever opens it. `target` is a file
    in a directory the policy never sees, replaced whole with the last `limit` bytes of the log
    whenever the log changes, read only from a regular file without following a link; it trails
    the log by up to `interval_s`, catches up once more on the way out, and is removed then. A
    kept server's log holds earlier units too: only what it wrote from byte `start` on is copied."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")
    seen: Any = _copy_tail(source, target, limit, None, start)
    stop = threading.Event()

    def follow() -> None:
        nonlocal seen
        while not stop.wait(interval_s):
            seen = _copy_tail(source, target, limit, seen, start)

    thread = threading.Thread(target=follow, name="vector-policy-log", daemon=True)
    thread.start()
    try:
        yield target
    finally:
        stop.set()
        thread.join()
        target.unlink(missing_ok=True)


def _alive() -> PolicyEnd | None:
    return None


@dataclass(frozen=True)
class ServedPolicy:
    """One unit's policy, listening.

    The benchmark subprocess gets `address` and the *name* `authkey_env`; `env` is what must be
    added to its environment for that name to hold the key. `log_file` is the server's log (and
    everything the policy printed), kept in the unit's directory once the unit is over; `live_log`
    is that log as the policy writes it while it serves, which only `mirror_log` reads.
    """

    address: str
    authkey_env: str
    env: dict[str, str]
    log_file: Path
    #: Called once the unit is over: how the policy ended underneath it, or None if it did not.
    died: Callable[[], PolicyEnd | None] = _alive
    #: The log the policy writes while it serves; None where the runtime has none.
    live_log: Path | None = None
    #: The policy server's process group on this host, whose CPU counts as the unit's progress for
    #: the stall watchdog (a benchmark waiting on a policy that is building its model is not
    #: hung); None where the runtime cannot say.
    pgid: int | None = None
    #: Where this unit's part of `live_log` starts: a kept server's log holds earlier units too.
    log_start: int = 0


@dataclass
class _Server:
    """One `vector_protocol.serve` process, listening."""

    process: subprocess.Popen
    sockets: Path
    address: Path
    log_file: Path
    key: bytes
    ledger: Ledger
    #: What it serves, and on what: a kept server is reused only for the same of both.
    handle: str
    env: tuple[tuple[str, str], ...]

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self, log_to: Path | None = None) -> None:
        """Kill its group, then copy its log to `log_to` (when given) before its directory goes."""
        _kill_group(self.process)
        self.ledger.ended(self.process.pid)
        if log_to is not None:
            copy_log(self.log_file, log_to, MAX_LOG_BYTES)
        shutil.rmtree(self.sockets, ignore_errors=True)


class ServerPool:
    """A policy server kept between units, one at a time: a slot's (`serve(pool=...)`).

    `ledger_dir` is where the kept server's process group is recorded while it runs, for the
    reaper of a killed orchestrator (`orphans`)."""

    def __init__(self, ledger_dir: Path) -> None:
        self.ledger_dir = Path(ledger_dir)
        self.current: _Server | None = None

    @property
    def handle(self) -> str | None:
        """What the kept server serves, if one is kept."""
        return self.current.handle if self.current is not None else None

    def close(self) -> None:
        if self.current is not None:
            self.current.stop()
            self.current = None


class SubprocessPolicyRuntime:
    """Serve local directories' policies as subprocesses."""

    #: Variables a served policy inherits beyond the subprocess allow-list (`benchmark_environment`).
    env_keep: tuple[str, ...] = ()
    #: Variables set for a served policy whatever the host has (a thread cap, say).
    env_set: dict[str, str] = {}

    def __init__(
        self,
        spec: Any,
        local: Mapping[str, str | os.PathLike[str]],
        *,
        python: str = sys.executable,
        start_timeout_s: float | None = None,
    ) -> None:
        self.spec = spec
        self.local = local_directories(local)
        self.python = python
        self.start_timeout_s = float(
            start_timeout_s if start_timeout_s is not None else spec.budgets["policy_start"]
        )

    # -- the seam ---------------------------------------------------------------------------

    def directory(self, ref_or_repo: SubmissionRef | str) -> Path:
        repo = ref_or_repo if isinstance(ref_or_repo, str) else ref_or_repo.repo
        key = None if isinstance(ref_or_repo, str) else ref_or_repo.key
        found = self.local.get(repo) or (self.local.get(key) if key else None)
        if found is None:
            raise RuntimeUnavailable(f"no local directory for {repo}")
        if not found.is_dir():
            raise RuntimeUnavailable(f"{found} is not a directory")
        return found

    def resolve(self, repo: str, revision: str) -> SubmissionRef:
        """A commit sha as given (the directory stands for that commit), or the directory's tree
        hash for a branch or tag name, which a local directory cannot resolve."""
        directory = self.directory(repo)
        sha = revision if is_commit_sha(revision) else tree_hash(directory)
        return SubmissionRef.resolved(repo, sha)

    def fetch(self, ref: SubmissionRef, *, workdir: Path) -> FetchedSubmission:
        directory = self.directory(ref)
        return FetchedSubmission(ref=ref, commit=ref.revision, root=directory)

    def prepare(self, fetched: FetchedSubmission, *, workdir: Path) -> PreparedSubmission:
        path = fetched.root / MANIFEST
        try:
            manifest = load_manifest(path)
        except ManifestError as exc:
            raise SubmissionRefused("manifest", f"{MANIFEST}: {'; '.join(exc.problems)}") from None
        if manifest.api != MANIFEST_API:
            raise SubmissionRefused(
                "manifest", f"{MANIFEST}: api {manifest.api} is not {MANIFEST_API}"
            )
        try:
            with self._serve(path, Path(workdir)) as (served, key, live_log):
                with RemotePolicy(
                    served.address, key, timeout_s=self.start_timeout_s, log_file=live_log
                ) as policy:
                    policy.hello()
        except PolicyDied as exc:
            raise SubmissionRefused("start", str(exc)) from None
        except PolicyUnavailable as exc:
            raise SubmissionRefused("hello", str(exc)) from None
        return PreparedSubmission(
            ref=fetched.ref, commit=fetched.commit, weights_sha256=None, handle=path
        )

    @contextmanager
    def serve(
        self,
        prepared: PreparedSubmission,
        *,
        workdir: Path,
        env: Mapping[str, str] | None = None,
        pool: ServerPool | None = None,
    ) -> Iterator[ServedPolicy]:
        if pool is None:
            with self._serve(Path(prepared.handle), Path(workdir), env) as (served, _, _live):
                yield served
            return
        yield from self._serve_kept(prepared, Path(workdir), env, pool)

    # -- a server ---------------------------------------------------------------------------

    def _launch(
        self, manifest: Path, ledger_dir: Path, env: Mapping[str, str] | None, *, keep: bool
    ) -> _Server:
        """One `vector_protocol.serve` process, listening; `PolicyDied` if it never does.

        Its log is written beside its socket, in a temporary directory of its own: nothing on the
        policy's command line names the run directory, which would tell it which side it is on
        and where the other side's results are."""
        ledger_dir.mkdir(parents=True, exist_ok=True)
        # A Unix socket's path is limited to about 100 bytes, which a run directory can exceed.
        sockets = Path(tempfile.mkdtemp(prefix="vector-duel-"))
        address = sockets / SOCKET_FILE
        log_file = sockets / LOG_FILE
        key = secrets.token_bytes(AUTHKEY_BYTES)
        extra = dict(env or {})
        environ = {
            **benchmark_environment(os.environ, AUTHKEY_ENV, self.env_keep),
            **self.env_set,
            **extra,
            AUTHKEY_ENV: key.hex(),
        }
        argv = [
            self.python,
            "-m",
            "vector_protocol.serve",
            "--manifest",
            str(manifest),
            "--address",
            str(address),
            "--authkey-env",
            AUTHKEY_ENV,
            "--log-file",
            str(log_file),
            *(["--keep"] if keep else []),
        ]
        try:
            process = subprocess.Popen(
                argv,
                env=environ,
                cwd=sockets,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            shutil.rmtree(sockets, ignore_errors=True)
            raise RuntimeUnavailable(f"the policy server could not be started: {exc}") from None
        server = _Server(
            process=process,
            sockets=sockets,
            address=address,
            log_file=log_file,
            key=key,
            ledger=Ledger(ledger_dir),
            handle=str(manifest),
            env=tuple(sorted(extra.items())),
        )
        try:
            server.ledger.started(process.pid)
            self._wait_listening(process, address, log_file)
        except BaseException:
            server.stop()
            raise
        return server

    def _served(self, server: _Server, workdir: Path, died: Any, start: int = 0) -> ServedPolicy:
        return ServedPolicy(
            address=str(server.address),
            authkey_env=AUTHKEY_ENV,
            env={AUTHKEY_ENV: server.key.hex()},
            log_file=workdir / LOG_FILE,
            died=died,
            live_log=server.log_file,
            pgid=server.process.pid,
            log_start=start,
        )

    @contextmanager
    def _serve(
        self, manifest: Path, workdir: Path, env: Mapping[str, str] | None = None
    ) -> Iterator[tuple[ServedPolicy, bytes, Path]]:
        """A server for one client, killed with its group on the way out; its log is copied into
        `workdir` when it is over."""
        workdir.mkdir(parents=True, exist_ok=True)
        server = self._launch(manifest, workdir, env, keep=False)
        try:
            served = self._served(
                server, workdir, lambda: self._died(server.process, server.log_file)
            )
            self._started(server.process, served)
            yield served, server.key, server.log_file
        finally:
            server.stop(log_to=workdir / LOG_FILE)

    def _serve_kept(
        self,
        prepared: PreparedSubmission,
        workdir: Path,
        env: Mapping[str, str] | None,
        pool: ServerPool,
    ) -> Iterator[ServedPolicy]:
        """One unit served by the pool's kept server, started - or started again, for another
        submission or card, or after it died - when it must be. The unit's part of its log is
        copied into `workdir` when the unit is over."""
        workdir.mkdir(parents=True, exist_ok=True)
        wanted = (str(prepared.handle), tuple(sorted(dict(env or {}).items())))
        kept = pool.current
        if kept is not None and ((kept.handle, kept.env) != wanted or not kept.alive()):
            pool.close()
            kept = None
        # A server started for this unit gives it its whole log, the model being built included.
        start = _size(kept.log_file) if kept is not None else 0
        if kept is None:
            kept = self._launch(Path(prepared.handle), pool.ledger_dir, env, keep=True)
            pool.current = kept
        served = self._served(kept, workdir, lambda: _ended(kept), start)
        self._started(kept.process, served)
        try:
            yield served
        finally:
            copy_log(kept.log_file, workdir / LOG_FILE, MAX_LOG_BYTES, start=start)
            if not kept.alive():
                pool.close()

    def _wait_listening(self, process: subprocess.Popen, address: Path, log_file: Path) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        while not address.exists():
            code = process.poll()
            if code is not None:
                raise PolicyDied(
                    f"the policy exited ({code}) before listening\n{tail(log_file)}".rstrip()
                )
            if time.monotonic() >= deadline:
                raise PolicyDied(
                    f"the policy did not listen within {self.start_timeout_s:g}s\n"
                    f"{tail(log_file)}".rstrip()
                )
            time.sleep(POLL_S)

    def _started(self, process: subprocess.Popen, served: ServedPolicy) -> None:
        """Called once a server listens. The tests stand in here to kill one."""

    @staticmethod
    def _died(process: subprocess.Popen, log_file: Path) -> PolicyEnd | None:
        """How the server ended underneath its unit: any end but a clean exit after its client.
        It is killed only on the way out of `_serve`, after this is read, so an end seen here is
        the policy's own - on this host nothing tells a policy that exited from one killed by the
        machine's own OOM killer."""
        try:
            code = process.wait(timeout=EXIT_GRACE_S)
        except subprocess.TimeoutExpired:
            return None  # still serving a client that never came or never left: not dead
        if code == 0:
            return None
        how = f"was killed by signal {-code}" if code < 0 else f"exited {code}"
        return PolicyEnd(f"the policy process {how}\n{tail(log_file)}".rstrip(), POLICY)


def _size(path: Path) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _ended(server: _Server) -> PolicyEnd | None:
    """How a kept server ended underneath its unit: a non-zero exit or a signal. Still serving, or
    gone with its client (a hang-up during a call exits 0), is not the policy's doing."""
    code = server.process.poll()
    if code is None or code == 0:
        return None
    how = f"was killed by signal {-code}" if code < 0 else f"exited {code}"
    return PolicyEnd(f"the policy process {how}\n{tail(server.log_file)}".rstrip(), POLICY)


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.kill()
    process.wait()
