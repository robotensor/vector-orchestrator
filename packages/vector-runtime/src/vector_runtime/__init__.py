"""The validator's own runtime for Vector (Vector policy) submissions.

A miner submits weights only: one `model.safetensors`, no code and no pickle. Everything that
turns those bytes into a policy is the validator's:

- `vector_runtime.check`: is this file exactly the pinned architecture's tensors? Header only, the
  standard library only, so it runs on a host without torch.
- `vector_runtime.template`: the pinned architecture, `arch/<name>.cfg.json` (the resolved model and
  prompt-dataset configuration) and `arch/<name>.tensors.json` (every tensor's shape and dtype).
- `vector_runtime.policy.VectorPolicy`: builds the model from the template, loads the weights and
  serves it over the `vector-protocol` protocol.
  the only code that unpickles, and only a file its caller owns.

Importing this package imports nothing but the standard library.
"""

__version__ = "0.1.0.dev0"

#: The architecture a submission is held to, and the name of its template files.
ARCHITECTURE = "vector_robotwin_l1_v1"
#: The one file a submission holds.
WEIGHTS_FILENAME = "model.safetensors"

__all__ = ["ARCHITECTURE", "WEIGHTS_FILENAME", "__version__"]
