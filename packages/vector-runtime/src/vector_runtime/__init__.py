"""The validator's runtime for `vector_v1.1` weights.

A submission is one `model.safetensors`: no code, no pickle. `check` holds its header to the
architecture's tensors (standard library only), `net` builds the network and loads it, and
`policy.VectorPolicy` serves it over `vector-protocol`.
"""

__version__ = "0.3.0"

ARCHITECTURE = "vector_v1.1"
WEIGHTS_FILENAME = "model.safetensors"
