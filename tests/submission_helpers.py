"""A stand-in for the Hub, and a policy repository written on disk, for the pure suite."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import huggingface_hub
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

#: The HTTP client huggingface_hub raises its errors with: httpx2 from its version 2, httpx before.
httpx = importlib.import_module(
    "httpx2" if int(huggingface_hub.__version__.split(".")[0]) >= 2 else "httpx"
)

SHA_A = "a" * 40
SHA_B = "b" * 40


def hub_response(status: int) -> Any:
    """A response as huggingface_hub's errors need one, with its request attached."""
    request = httpx.Request("GET", "https://huggingface.co/api/models/org/policy")
    return httpx.Response(status, request=request)


def not_found(kind, what: str, detail: str, repo: str, revision: str):
    """The 404 the Hub answers with, in the shape huggingface_hub raises it: a request id first,
    what was not found lines later, and the server's own words last and in `server_message`."""
    message = (
        "404 Client Error. (Request ID: Root=1-6aa6e45a-5ab94e211267ab2143004ac4)\n\n"
        f"{what} Not Found for url: https://huggingface.co/api/models/{repo}/revision/{revision}"
        f"?blobs=true.\n{detail}"
    )
    return kind(message, response=hub_response(404), server_message=detail)


@dataclass
class FakeSibling:
    rfilename: str
    size: int | None


@dataclass
class FakeInfo:
    sha: str
    siblings: list[FakeSibling]


@dataclass
class FakeRef:
    name: str
    ref: str
    target_commit: str


@dataclass
class FakeRefs:
    branches: list[FakeRef]
    tags: list[FakeRef]
    pull_requests: list[FakeRef] | None


@dataclass
class FakeCommit:
    commit_id: str


@dataclass
class FakeHub:
    """`HfApi.repo_info` over a table of repositories: `{repo: {revision: (sha, files)}}`, where
    `files` maps a path to its size. Every commit sha is also a revision of its repository.

    `list_repo_refs` and `list_repo_commits` answer from what `add` was told: each name is a branch
    at the commit, each of `tags` a tag, and `pr=N` makes it `refs/pr/N`'s alone. A commit added
    with none of them is the tip of a branch of its own, so it can be queued. `parent` puts the
    commit's history behind it: a commit added under a branch's name again moves the branch on and
    keeps the one before in its history."""

    repos: dict[str, dict[str, tuple[str, dict[str, int | None]]]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    trees: dict[str, Path] = field(default_factory=dict)
    #: `{repo: {name: sha}}` for branches and tags, `{repo: {number: sha}}` for pull requests.
    branches: dict[str, dict[str, str]] = field(default_factory=dict)
    tags: dict[str, dict[str, str]] = field(default_factory=dict)
    pull_requests: dict[str, dict[int, str]] = field(default_factory=dict)
    #: `{repo: {sha: [sha, parent, grandparent, ...]}}`, as the Hub lists a revision's commits.
    history: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    ref_calls: list[str] = field(default_factory=list)
    commit_calls: list[tuple[str, str]] = field(default_factory=list)

    def add(
        self,
        repo: str,
        sha: str,
        files: dict[str, int | None],
        *names: str,
        tags: tuple[str, ...] = (),
        pr: int | None = None,
        parent: str | None = None,
    ) -> None:
        entry = self.repos.setdefault(repo, {})
        entry[sha] = (sha, files)
        history = self.history.setdefault(repo, {})
        history[sha] = [sha, *history.get(parent, [parent])] if parent else [sha]
        if pr is not None:
            assert not names and not tags, "a pull request's commit is on no branch or tag"
            entry[f"refs/pr/{pr}"] = (sha, files)
            self.pull_requests.setdefault(repo, {})[pr] = sha
            return
        for name in (*names, *tags):
            entry[name] = (sha, files)
        branches = self.branches.setdefault(repo, {})
        for name in names or (() if tags else (f"unnamed-{sha[:12]}",)):
            branches[name] = sha
        for name in tags:
            self.tags.setdefault(repo, {})[name] = sha

    def repo_info(self, repo_id, *, revision=None, repo_type=None, files_metadata=False):
        self.calls.append((repo_id, revision))
        assert repo_type == "model" and files_metadata
        if repo_id not in self.repos:
            raise not_found(
                RepositoryNotFoundError, "Repository", "Repository not found", repo_id, revision
            )
        if revision not in self.repos[repo_id]:
            raise not_found(
                RevisionNotFoundError, "Revision", f"Invalid rev id: {revision}", repo_id, revision
            )
        sha, files = self.repos[repo_id][revision]
        return FakeInfo(sha, [FakeSibling(p, s) for p, s in files.items()])

    def list_repo_refs(self, repo_id, *, repo_type=None, include_pull_requests=False):
        self.ref_calls.append(repo_id)
        assert repo_type == "model"
        if repo_id not in self.repos:
            raise not_found(
                RepositoryNotFoundError, "Repository", "Repository not found", repo_id, "refs"
            )

        def refs(table: dict, prefix: str) -> list[FakeRef]:
            return [FakeRef(str(n), f"{prefix}{n}", s) for n, s in table.get(repo_id, {}).items()]

        return FakeRefs(
            branches=refs(self.branches, "refs/heads/"),
            tags=refs(self.tags, "refs/tags/"),
            pull_requests=refs(self.pull_requests, "refs/pr/") if include_pull_requests else None,
        )

    def list_repo_commits(self, repo_id, *, repo_type=None, revision=None):
        self.commit_calls.append((repo_id, revision))
        assert repo_type == "model"
        if repo_id not in self.repos:
            raise not_found(
                RepositoryNotFoundError, "Repository", "Repository not found", repo_id, revision
            )
        history = self.history.get(repo_id, {})
        if revision not in history:
            raise not_found(
                RevisionNotFoundError, "Revision", f"Invalid rev id: {revision}", repo_id, revision
            )
        return [FakeCommit(sha) for sha in history[revision]]


def write_policy_repo(root: Path, policy: str = "pkg.policy:Policy", **manifest: Any) -> Path:
    """A minimal competitor repository at `root`: a manifest and the package it names."""
    root.mkdir(parents=True, exist_ok=True)
    module = policy.partition(":")[0].split(".")
    package = root.joinpath(*module[:-1])
    package.mkdir(parents=True, exist_ok=True)
    for parent in [package, *package.parents]:
        if parent == root:
            break
        (parent / "__init__.py").touch()
    (package / f"{module[-1]}.py").write_text(
        "class Policy:\n"
        "    action_type = 'qpos'\n"
        "    def reset(self, seed): pass\n"
        "    def set_demonstration(self, arrays, info): pass\n"
        "    def act(self, observation): return {'action': [0.0]}\n"
    )
    lines = ["api: 1", f"policy: {policy}"]
    for key, value in manifest.items():
        lines.append(f"{key}: {value}")
    (root / "policy.yaml").write_text("\n".join(lines) + "\n")
    return root
