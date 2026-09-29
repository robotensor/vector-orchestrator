"""A submission: a Hugging Face model repository holding `model.safetensors` and at most a README.

Nothing in it is code, and nothing of it is executed or unpickled: the weights file's header is
checked against the architecture's tensors before a tensor is read, and the policy that loads it is
the validator's own (`vector_runtime.policy:VectorPolicy`, in the policy environment `python`).

- `resolve` pins `repo@revision` to a commit on the Hub, which also lists the repository's files.
- `fetch` refuses a repository holding anything else or no weights file, and downloads the weights
  file alone into a cache addressed by the commit.
- `prepare` checks the file (`vector_runtime.check`), writes a manifest naming the policy class with
  the weights' absolute path, and has the policy answer `hello`, which loads the weights: a file
  that passes the header check but does not load is refused before a unit is played.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vector_protocol.client import RemotePolicy
from vector_protocol.errors import PolicyUnavailable

from ..ids import SubmissionRef
from ..submissions import SubmissionError, SubmissionRejected
from ..submissions.fetch import HubFetcher, RepoCache
from ..submissions.resolve import REPO_TYPE, Resolved
from .runtime import (
    FetchedSubmission,
    PolicyDied,
    PreparedSubmission,
    RuntimeUnavailable,
    SubmissionRefused,
    SubprocessPolicyRuntime,
)

WEIGHTS_FILE = "model.safetensors"
ALLOWED_FILES = frozenset({WEIGHTS_FILE, "README.md", ".gitattributes"})
MAX_REPO_BYTES = 8 << 30
POLICY = "vector_runtime.policy:VectorPolicy"
#: Variables a served policy may use beyond the subprocess allow-list: a host's cache locations.
POLICY_ENV_KEEP = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TORCH_HOME", "XDG_CACHE_HOME")


def check_weights(weights: Path) -> Any:
    """`vector_runtime.check` on the file: the header against the architecture, and its sha256.
    Imported lazily: it is all of the runtime a validator host needs, and it needs no torch."""
    from vector_runtime.check import check

    return check(weights)


def architecture_sha256() -> tuple[str, str]:
    """The installed runtime's architecture and the sha256 of its tensor manifest."""
    from vector_runtime import ARCHITECTURE
    from vector_runtime.check import manifest

    return ARCHITECTURE, hashlib.sha256(manifest().read_bytes()).hexdigest()


class WeightsPolicyRuntime(SubprocessPolicyRuntime):
    """Serve weights-only submissions with the validator's policy class."""

    env_keep = POLICY_ENV_KEEP

    def __init__(
        self,
        spec: Any,
        *,
        python: str = sys.executable,
        cache_root: str | os.PathLike[str],
        api: Any = None,
        download: Callable[..., Any] | None = None,
        checker: Callable[[Path], Any] = check_weights,
        kwargs: dict[str, Any] | None = None,
        start_timeout_s: float | None = None,
        hello: bool = True,
        threads: int | None = 4,
    ) -> None:
        super().__init__(spec, {}, python=python, start_timeout_s=start_timeout_s)
        # Torch builds the model on this many CPU threads: uncapped, several servers starting
        # beside several simulators build an order of magnitude slower (12 s alone, 90 s four at once).
        if threads:
            self.env_set = {
                n: str(int(threads))
                for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            }
        self.cache = RepoCache(cache_root, MAX_REPO_BYTES)
        self.fetcher = HubFetcher(self.cache, api=api, download=download or self._download)
        self.checker = checker
        #: Extra keywords for the policy class (a device, say); never the weights path.
        self.kwargs = dict(kwargs or {})
        self.hello = hello

    def resolve(self, repo: str, revision: str) -> SubmissionRef:
        with _mapped():
            return self.fetcher.resolve(repo, revision).ref

    def fetch(self, ref: SubmissionRef, *, workdir: Path) -> FetchedSubmission:
        with _mapped():
            resolved = self.fetcher.resolve(ref.repo, ref.revision)
            names = sorted(f.path for f in resolved.files)
            extra = [name for name in names if name not in ALLOWED_FILES]
            if extra:
                shown = ", ".join(extra[:5]) + (", ..." if len(extra) > 5 else "")
                raise SubmissionRejected(
                    "fetch",
                    f"{ref.entry} holds files a submission may not ({shown}): only {WEIGHTS_FILE} and a README",
                )
            if WEIGHTS_FILE not in names:
                raise SubmissionRejected("fetch", f"{ref.entry} holds no {WEIGHTS_FILE}")
            weights_only = Resolved(
                repo=resolved.repo,
                revision=resolved.revision,
                sha=resolved.sha,
                files=tuple(f for f in resolved.files if f.path == WEIGHTS_FILE),
            )
            fetched = self.fetcher.fetch(weights_only)
        return FetchedSubmission(ref=ref, commit=resolved.sha, root=fetched.root)

    def prepare(self, fetched: FetchedSubmission, *, workdir: Path) -> PreparedSubmission:
        try:
            installed = architecture_sha256()
        except Exception as exc:  # noqa: BLE001 - a runtime that cannot say is the harness's
            raise RuntimeUnavailable(f"vector_runtime cannot be read: {exc}") from exc
        pinned = (self.spec.model["architecture"], self.spec.model["tensors_sha256"])
        if installed != pinned:
            raise RuntimeUnavailable(
                f"vector_runtime holds {installed[0]} ({installed[1][:12]}), the spec pins {pinned[0]} ({pinned[1][:12]})"
            )
        weights = (fetched.root / WEIGHTS_FILE).resolve()
        if not weights.is_file():
            raise SubmissionRefused("fetch", f"{WEIGHTS_FILE} is not in the fetched checkout")
        try:
            report = self.checker(weights)
        except Exception as exc:  # noqa: BLE001 - a checker that cannot run is the harness's
            raise RuntimeUnavailable(f"the weights check could not run: {exc}") from exc
        if not report.ok:
            raise SubmissionRefused(
                "check", f"{WEIGHTS_FILE}: " + ("; ".join(report.errors[:8]) or "no reason given")
            )
        manifest = self._write_manifest(Path(workdir), weights)
        if self.hello:
            try:
                with self._serve(manifest, Path(workdir)) as (served, key, live_log):
                    with RemotePolicy(
                        served.address, key, timeout_s=self.start_timeout_s, log_file=live_log
                    ) as policy:
                        policy.hello()
            except PolicyDied as exc:
                raise SubmissionRefused("start", str(exc)) from None
            except PolicyUnavailable as exc:
                raise SubmissionRefused("hello", str(exc)) from None
        return PreparedSubmission(
            ref=fetched.ref,
            commit=fetched.commit,
            weights_sha256=report.weights_sha256,
            handle=manifest,
        )

    def _write_manifest(self, workdir: Path, weights: Path) -> Path:
        """The validator's own `policy.yaml` (JSON is valid YAML), with the weights' absolute path."""
        directory = (workdir / "manifest").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "policy.yaml"
        doc = {"api": 1, "policy": POLICY, "kwargs": {**self.kwargs, "weights": str(weights)}}
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return path

    def _download(self, repo: str, *, revision: str, repo_type: str, local_dir: str) -> None:
        from huggingface_hub import hf_hub_download

        if repo_type != REPO_TYPE:
            raise SubmissionError(f"a submission is a {REPO_TYPE} repository, not a {repo_type}")
        hf_hub_download(
            repo, WEIGHTS_FILE, revision=revision, repo_type=repo_type, local_dir=local_dir
        )


@contextmanager
def _mapped() -> Iterator[None]:
    """The Hub code's two errors as the runtime's."""
    try:
        yield
    except SubmissionRejected as exc:
        raise SubmissionRefused(exc.step, exc.reason) from None
    except SubmissionError as exc:
        raise RuntimeUnavailable(str(exc)) from None
