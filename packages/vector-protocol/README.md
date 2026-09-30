# robotensor-vector-protocol

Policy protocol for Robotensor Vector. The benchmark drives a policy it never imports: the policy
runs in its own process, and the two exchange named numpy arrays and JSON fields over an
authenticated socket. Nothing is pickled. Depends only on numpy and PyYAML.

## Policy

A policy repo has a `policy.yaml` at its root:

```yaml
api: 1
policy: my_policy.policy:MyPolicy   # module:Class
kwargs: {checkpoint: weights.pt}    # optional
requirements: requirements.txt      # optional
```

```python
class MyPolicy:
    action_type = "qpos"  # or "ee"

    def set_demonstration(self, arrays, info): ...
    def reset(self, seed: int): ...
    def act(self, observation) -> dict: ...  # {"action": (A,) or (H, A)}
    def close(self): ...  # optional
```

See `examples/replay_policy` and `examples/zero_policy`.

## Serving

```bash
export VECTOR_AUTHKEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python -m vector_protocol.serve --manifest policy.yaml \
    --address /tmp/policy.sock --authkey-env VECTOR_AUTHKEY
```

`--address` is a Unix socket path or `host:port`. The key is removed from the environment before
policy code runs.

## Client

```python
from vector_protocol.client import RemotePolicy

with RemotePolicy(address, authkey, timeout_s=60.0) as policy:
    policy.hello()
    policy.set_demonstration(arrays, info)
    policy.reset(seed)
    action = policy.act(observation)["action"]
```

Any failure raises `vector_protocol.PolicyUnavailable` with the tail of the server log.
