"""A duel: both sides of it on the same units and the same prompt bytes, scored and published.

- `runtime`, `weights_runtime`: serving a side's policy, one fresh server per unit.
- `materialize`: every unit's prompt, produced once, before either side runs.
- `side`: one side's units, one served policy per unit, resumable from its results file.
- `score`: per-task rates, the crown rule and void units.
- `orchestrate`: all of it, from a request to a published record.
"""
