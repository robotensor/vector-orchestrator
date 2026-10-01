# vector-orchestrator

Duel engine for **Robotensor Vector**. A challenger plays the reigning king on the same units and
the same demonstrations. The result is written to an append-only store and mirrored to a Hugging
Face dataset. The subnet validator ([robotensor-subnet](https://github.com/robotensor/robotensor-subnet))
drives it; `vector-orchestrator duel` runs one by hand.

## Layout

| Path | |
|---|---|
| `spec.json` | The contract: benchmark checkout, level, the 16 tasks, runs per task, step ratio, crown rule, architecture, time budgets |
| `src/vector_orchestrator/` | Units, duels, scoring, submission fetch, result store |
| `packages/vector-runtime` | Checks, loads and serves `vector_v1.1` weights |
| `packages/vector-protocol` | Process-isolated policy protocol (numpy + JSON over an authenticated socket) |

## How a duel works

- **Units.** Each task gets `runs_per_task` units. A unit pairs a demonstration scene with a scoring
  scene, both drawn from a hash of the duel id and the seeding chain block.
- **Benchmark.** [RoboTwin-Vector](https://github.com/robotensor/RoboTwin-Vector) runs from its
  checkout under the simulator's interpreter and is never imported.
- **Submissions.** A Hugging Face repo with `model.safetensors` only. The weights are checked
  against the architecture and served by the validator's own code; no submitted code runs.
- **Scoring.** A side's score is its mean success rate over tasks, counting non-void units only.
  The challenger is crowned when it leads by `crown.margin` and passes a one-sided sign test at
  `crown.alpha`. A duel with more than `crown.max_void` void units is void.
- **Hardware.** Uses every visible GPU, `--workers` units per card. Interrupted duels resume from
  `runs/<event id>/`.

## Usage

```bash
uv venv --python 3.10 .venv
uv pip install -e ".[dev]" -e packages/vector-protocol -e packages/vector-runtime

export ROBOTWIN_BENCH_ROOT=/path/to/RoboTwin-Vector
export ROBOTWIN_BENCH_PYTHON=/path/to/robotwin/bin/python

vector-orchestrator check
vector-orchestrator store init --root store/
vector-orchestrator duel --store store/ --run-dir runs/ --challenger owner/name@main \
    --policy-python /path/to/policy/bin/python --policy-kwarg device=cuda:0 --workers 1
vector-orchestrator store mirror --root store/ --repo owner/results
```

## Store format

```
manifest.json              schema and spec version
head.json                  current seq, block and king
queue.json                 validator queue
running.json               the duel running now and how far it has got
champions.json             the models the validator pays now
index-NNNN.jsonl           one line per duel
events/<event_id>.json     units, per-task scores, verdict, seed block
media/<sha[:2]>/<sha>.mp4  demonstration and rollout clips
```

## Tests

```bash
ruff check . && ruff format --check .
pytest
pytest packages/vector-runtime -m "not gpu"
```

## License

Apache-2.0
