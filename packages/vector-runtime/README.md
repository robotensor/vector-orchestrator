# robotensor-vector-runtime

Validator runtime for Robotensor Vector submissions. A submission is a single `model.safetensors`
of the `vector_v1.1` architecture: no code, no pickle. The runtime checks the header against the
architecture, builds the network with its own code, loads the weights, and serves the policy over
`vector-protocol`.

`vector_v1.1` is a Vector policy for two Franka arms: CLIP ViT-B/16 encoder, 6-layer
prompt decoder, 1-D diffusion U-Net; 753 tensors, 781M parameters
(`src/vector_runtime/vector_v1.1.json`).

## Install

```bash
pip install robotensor-vector-runtime                                    # header check only, stdlib
pip install "robotensor-vector-runtime[model]" robotensor-vector-protocol   # load and serve
```

## Usage

```bash
vector-runtime check   --weights DIR_OR_FILE   # prints weights_sha256
```

To serve, point a `policy.yaml` at `VectorPolicy`:

```yaml
api: 1
policy: vector_runtime.policy:VectorPolicy
kwargs: {weights: /abs/path/model.safetensors, device: "cuda:0"}
```
