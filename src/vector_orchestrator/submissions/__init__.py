"""A submission's repository on the Hub: `resolve` pins `repo@revision` to its commit, once, at
queue time, and `fetch` downloads it into a cache addressed by that commit."""

from .errors import SubmissionError, SubmissionRejected

__all__ = ["SubmissionError", "SubmissionRejected"]
