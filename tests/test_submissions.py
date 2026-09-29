"""Resolving, fetching and checking a submission, with the Hub stood in for."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from huggingface_hub.errors import HfHubHTTPError

from submission_helpers import SHA_A, SHA_B, FakeHub, httpx, hub_response, write_policy_repo
from vector_orchestrator.duel.weights_runtime import MAX_REPO_BYTES
from vector_orchestrator.ids import SubmissionRef
from vector_orchestrator.submissions import SubmissionError, SubmissionRejected
from vector_orchestrator.submissions.fetch import HubFetcher, RepoCache, measure
from vector_orchestrator.submissions.resolve import Resolved, resolve

# -- resolve ------------------------------------------------------------------------------------


@pytest.fixture
def hub():
    hub = FakeHub()
    hub.add("org/policy", SHA_A, {"policy.yaml": 80, "policy.py": 1200, "weights.pt": 4000}, "main")
    hub.add("org/policy", SHA_B, {"policy.yaml": 80, "policy.py": 1300}, "v2", "dev")
    return hub


def test_a_branch_or_tag_resolves_to_its_commit_and_a_sha_to_itself(hub):
    assert resolve("org/policy", "main", api=hub).sha == SHA_A
    assert resolve("org/policy", "v2", api=hub).sha == SHA_B
    assert resolve("org/policy", "dev", api=hub).sha == SHA_B
    assert resolve("org/policy", SHA_A, api=hub).sha == SHA_A
    # The Hub was asked exactly once each: resolution happens at queue time and never again.
    assert hub.calls == [("org/policy", r) for r in ("main", "v2", "dev", SHA_A)]


def test_the_resolved_ref_is_keyed_by_the_sha_and_lists_the_files_with_sizes(hub):
    resolved = resolve("org/policy", "main", api=hub)
    assert resolved == Resolved(repo="org/policy", revision="main", sha=SHA_A, files=resolved.files)
    assert resolved.ref == SubmissionRef.resolved("org/policy", SHA_A)
    assert resolved.ref.key == SubmissionRef.make("org/policy", SHA_A).key
    assert {f.path: f.size for f in resolved.files} == {
        "policy.yaml": 80,
        "policy.py": 1200,
        "weights.pt": 4000,
    }
    assert resolved.declared_bytes == 5280


def test_what_the_hub_does_not_have_is_rejected_with_the_reason(hub):
    """The reason carries the Hub's own words, which its 404 puts lines after the request id."""
    with pytest.raises(SubmissionRejected) as info:
        resolve("org/missing", "main", api=hub)
    assert info.value.step == "resolve"
    assert info.value.reason == "org/missing@main: repository not found"
    with pytest.raises(SubmissionRejected) as info:
        resolve("org/policy", "no-such-branch", api=hub)
    assert info.value.reason == (
        "org/policy@no-such-branch: revision not found: Invalid rev id: no-such-branch"
    )
    assert "Request ID" not in info.value.reason
    with pytest.raises(SubmissionRejected, match="revision not found: Invalid rev id: c+$"):
        resolve("org/policy", "c" * 40, api=hub)
    with pytest.raises(SubmissionRejected, match="is not a Hugging Face repo id"):
        resolve("not a repo", "main", api=hub)
    with pytest.raises(SubmissionRejected, match="no revision"):
        resolve("org/policy", "", api=hub)
    assert ("not a repo", "main") not in hub.calls


def test_a_hub_that_cannot_be_asked_is_the_harness_problem_not_the_submissions(hub):
    class Down:
        def repo_info(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    class Overloaded:
        def repo_info(self, *args, **kwargs):
            raise HfHubHTTPError("503 Server Error", response=hub_response(503))

    class Lying:
        def repo_info(self, *args, **kwargs):
            return type("Info", (), {"sha": "not-a-sha", "siblings": []})()

    class Swapping:
        def repo_info(self, *args, **kwargs):
            return type("Info", (), {"sha": SHA_B, "siblings": []})()

    with pytest.raises(SubmissionError, match="unreachable"):
        resolve("org/policy", "main", api=Down())
    with pytest.raises(SubmissionError, match="503"):
        resolve("org/policy", "main", api=Overloaded())
    with pytest.raises(SubmissionError, match="no commit sha"):
        resolve("org/policy", "main", api=Lying())
    with pytest.raises(SubmissionError, match="to another"):
        resolve("org/policy", SHA_A, api=Swapping())


def test_an_entry_the_hub_gives_no_size_for_counts_nothing_and_is_not_downloaded(
    hub, tmp_path, source
):
    """With `files_metadata` the Hub sizes every file, LFS ones included; an entry it does not
    size cannot be held to max_repo_bytes before it is on disk, so it is not fetched at all -
    the Hub's answer is the harness's problem, not the entry's."""
    hub.add("org/lfs", SHA_A, {"policy.yaml": 80, "weights.pt": None}, "main")
    resolved = resolve("org/lfs", "main", api=hub)
    assert resolved.declared_bytes == 80, "a lower bound: what was sized"
    assert [f.size for f in resolved.files] == [80, None]
    calls: list = []
    fetcher = HubFetcher(
        RepoCache(tmp_path / "cache", 10**9), api=hub, download=fake_download(source, calls)
    )
    with pytest.raises(SubmissionError, match="declared no size for 1 file.*weights.pt") as info:
        fetcher.fetch(resolved)
    assert not isinstance(info.value, SubmissionRejected)
    assert calls == [] and not (tmp_path / "cache" / SHA_A).exists()


# -- fetch --------------------------------------------------------------------------------------


def fake_download(source: Path, calls: list):
    """`snapshot_download` stood in for: copies `source` into `local_dir` and leaves the Hub's
    bookkeeping directory behind, as the real one does."""

    def download(repo_id, *, revision, repo_type, local_dir):
        calls.append((repo_id, revision, repo_type))
        shutil.copytree(source, local_dir, dirs_exist_ok=True)
        bookkeeping = Path(local_dir) / ".cache" / "huggingface" / "download"
        bookkeeping.mkdir(parents=True)
        (bookkeeping / "policy.py.metadata").write_text("etag")

    return download


@pytest.fixture
def source(tmp_path):
    root = write_policy_repo(tmp_path / "source")
    (root / "weights.bin").write_bytes(b"\0" * 4000)
    return root


def test_fetch_downloads_once_into_a_checkout_addressed_by_the_sha(spec, tmp_path, hub, source):
    calls: list = []
    cache = RepoCache(tmp_path / "cache", MAX_REPO_BYTES)
    fetcher = HubFetcher(cache, api=hub, download=fake_download(source, calls))
    resolved = fetcher.resolve("org/policy", "main")

    fetched = fetcher.fetch(resolved)
    assert fetched.root == tmp_path / "cache" / SHA_A / "repo"
    assert not fetched.cached and calls == [("org/policy", SHA_A, "model")]
    assert sorted(p.name for p in fetched.root.iterdir()) == ["pkg", "policy.yaml", "weights.bin"]
    assert not (fetched.root / ".cache").exists(), "the Hub's bookkeeping is not the repository"
    assert (fetched.bytes, fetched.files) == measure(source)
    marker = json.loads((tmp_path / "cache" / SHA_A / "fetched.json").read_text())
    assert (marker["repo"], marker["sha"], marker["bytes"]) == ("org/policy", SHA_A, fetched.bytes)

    again = fetcher.fetch(resolved)
    assert again.cached and again.root == fetched.root and len(calls) == 1
    # Another revision name for the same commit is the same checkout.
    hub.add("org/policy", SHA_A, {}, "release")
    assert fetcher.fetch(fetcher.resolve("org/policy", "release")).cached and len(calls) == 1


def test_fetch_refuses_a_repository_over_max_repo_bytes_before_and_after_download(
    tmp_path, hub, source
):
    calls: list = []
    # The Hub declares 5280 bytes for org/policy@main; a limit under that refuses it unfetched.
    small = HubFetcher(
        RepoCache(tmp_path / "small", 5000), api=hub, download=fake_download(source, calls)
    )
    with pytest.raises(
        SubmissionRejected, match="declares 5280 bytes, over max_repo_bytes"
    ) as info:
        small.fetch(small.resolve("org/policy", "main"))
    assert info.value.step == "fetch" and calls == []
    assert not (tmp_path / "small" / SHA_A).exists()

    # Declared sizes are what the Hub says; what lands on disk is measured again.
    hub.add("org/policy", SHA_B, {"policy.yaml": 1}, "tiny")
    lying = HubFetcher(
        RepoCache(tmp_path / "lying", 100), api=hub, download=fake_download(source, calls)
    )
    with pytest.raises(SubmissionRejected, match="bytes on disk, over max_repo_bytes"):
        lying.fetch(lying.resolve("org/policy", "tiny"))
    assert len(calls) == 1
    assert not (tmp_path / "lying" / SHA_B / "repo").exists()
    assert not (tmp_path / "lying" / SHA_B / "partial").exists()
    assert not (tmp_path / "lying" / SHA_B / "fetched.json").exists()


def test_a_download_that_fails_is_the_harness_problem_and_leaves_nothing_behind(
    spec, tmp_path, hub
):
    def broken(repo_id, **kwargs):
        Path(kwargs["local_dir"], "half.bin").write_bytes(b"x" * 10)
        raise HfHubHTTPError("502 Bad Gateway", response=hub_response(502))

    fetcher = HubFetcher(RepoCache(tmp_path / "cache", MAX_REPO_BYTES), api=hub, download=broken)
    with pytest.raises(SubmissionError, match="downloading org/policy@a+ failed: HfHubHTTPError"):
        fetcher.fetch(fetcher.resolve("org/policy", "main"))
    assert not (tmp_path / "cache" / SHA_A / "repo").exists()
    assert not (tmp_path / "cache" / SHA_A / "partial").exists()
    assert fetcher.cache.lookup(fetcher.resolve("org/policy", "main")) is None
