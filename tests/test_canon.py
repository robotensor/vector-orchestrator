from vector_orchestrator.canon import canonical_json, canonical_sha256, sha256_hex


def test_canonical_json_is_sorted_and_compact():
    assert (
        canonical_json({"b": 1, "a": [1, 2, {"z": None, "y": "é"}]})
        == '{"a":[1,2,{"y":"\\u00e9","z":null}],"b":1}'
    )


def test_sha256_matches_known_vector():
    assert sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert canonical_sha256({"a": 1}) == sha256_hex('{"a":1}')


def test_numbers_are_formatted_as_python_formats_them():
    """`3.0` is `3` to JSON.stringify and to RFC 8785, so a reader hashing a record hashes the line
    as published, never a re-encoding of it."""
    assert canonical_json({"score_margin": 3.0}) == '{"score_margin":3.0}'
    assert canonical_sha256({"score_margin": 3.0}) != sha256_hex('{"score_margin":3}')
