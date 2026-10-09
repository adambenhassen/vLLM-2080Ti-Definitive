# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the runtime rank-1 refusal projection."""

import importlib
import json
import struct

import numpy as np
import pytest
import torch

HIDDEN = 16
LAYERS = 4
STEM = "model.language_model.layers."


def _write_dirs(path, modules, seed=0):
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for m in modules:
        r = torch.randn(HIDDEN, generator=g)
        tensors[m] = r / r.norm()
    coefs = torch.linspace(0.9, 1.3, len(modules))
    header, blobs, off = {}, [], 0
    for name, t in [*tensors.items(), ("__coefs__", coefs)]:
        b = t.to(torch.float32).numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"coef_order": json.dumps(list(modules))}
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return tensors, {m: float(coefs[i]) for i, m in enumerate(modules)}


def _all_modules():
    out = []
    for i in range(LAYERS):
        attn = "self_attn.o_proj" if i % 4 == 3 else "linear_attn.out_proj"
        out += [f"{STEM}{i}.{attn}", f"{STEM}{i}.mlp.down_proj"]
    return out


@pytest.fixture
def rp(tmp_path, monkeypatch):
    """Fresh module state with a direction file covering every layer."""
    path = tmp_path / "dirs.safetensors"
    dirs, coefs = _write_dirs(path, _all_modules())
    monkeypatch.setenv("VLLM_REFUSAL_DIRS", str(path))
    monkeypatch.setenv("VLLM_REFUSAL_LAMBDA_INIT", "0.0")
    import vllm.refusal_projection as mod

    mod = importlib.reload(mod)
    mod.dirs_for_test, mod.coefs_for_test = dirs, coefs
    yield mod
    monkeypatch.delenv("VLLM_REFUSAL_DIRS")
    importlib.reload(mod)


def _claim_all(rp, prefix_fmt):
    got = {}
    for i in range(LAYERS):
        attn = "self_attn.o_proj" if i % 4 == 3 else "linear_attn.out_proj"
        for sub in (attn, "mlp.down_proj"):
            got[(i, sub)] = rp.resolve_direction(prefix_fmt.format(i), sub)
    return got


@pytest.mark.parametrize(
    "prefix_fmt", ["model.layers.{}", "language_model.model.layers.{}"]
)
def test_runtime_prefixes_claim_every_direction(rp, prefix_fmt):
    got = _claim_all(rp, prefix_fmt)
    assert all(v is not None for v in got.values())
    rp.verify_all_consumed()


def test_mtp_layers_are_never_projected(rp):
    assert rp.resolve_direction("mtp.layers.0", "mlp.down_proj") is None
    assert rp.resolve_direction("model.mtp.layers.0", "mlp.down_proj") is None


def test_unclaimed_direction_fails_closed(rp):
    rp.resolve_direction("model.layers.0", "mlp.down_proj")
    with pytest.raises(RuntimeError, match="not claimed"):
        rp.verify_all_consumed()


def test_double_claim_fails_closed(rp):
    rp.resolve_direction("model.layers.0", "mlp.down_proj")
    with pytest.raises(RuntimeError, match="claimed twice"):
        rp.resolve_direction("language_model.model.layers.0", "mlp.down_proj")


def test_lambda_zero_is_bit_exact_and_one_matches_weight_edit(rp):
    key = f"{STEM}0.mlp.down_proj"
    r, coef = rp.dirs_for_test[key], rp.coefs_for_test[key]
    proj = rp.RefusalProjection(r, coef, device=torch.device("cpu"))
    w = torch.randn(HIDDEN, 32, dtype=torch.float64)
    x = torch.randn(5, 32, dtype=torch.float64)
    y = (x @ w.T).to(torch.bfloat16)
    assert torch.equal(proj(y), y)

    rp.set_lambda(1.0)
    r64 = r.to(torch.float64)
    w_edit = w - coef * torch.outer(r64, r64) @ w
    want = x @ w_edit.T
    got = proj((x @ w.T).to(torch.float32)).to(torch.float64)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_per_token_buffer_drives_rows_and_set_lambda_mutates_in_place(rp):
    rp.ensure_token_buffer(8, torch.device("cpu"))
    key = f"{STEM}0.mlp.down_proj"
    r, coef = rp.dirs_for_test[key], rp.coefs_for_test[key]
    proj = rp.RefusalProjection(r, coef, device=torch.device("cpu"))
    assert proj.use_tok and proj.tok.data_ptr() == rp._tok_buf.data_ptr()

    y = torch.randn(3, HIDDEN) + 3 * r
    rp.fill_tokens(torch.tensor([0.0, 1.0, 0.0]), global_lambda=0.0)
    out = proj(y)
    assert torch.equal(out[0], y[0]) and torch.equal(out[2], y[2])
    assert abs(float(out[1] @ r) - float(y[1] @ r) * (1 - coef)) < 1e-4

    # Global lambda tensor is shared with the module, not copied.
    lam_ptr = proj.lam.data_ptr()
    rp.set_lambda(0.5)
    assert proj.lam.data_ptr() == lam_ptr and float(proj.lam) == 0.5


def test_fill_tokens_never_truncates(rp):
    rp.ensure_token_buffer(4, torch.device("cpu"))
    rp.fill_tokens(torch.ones(6), global_lambda=0.25)
    assert torch.all(rp._tok_buf == 0.25)


def test_compiled_forward_reads_live_buffers(rp):
    rp.ensure_token_buffer(8, torch.device("cpu"))
    key = f"{STEM}0.mlp.down_proj"
    proj = rp.RefusalProjection(
        rp.dirs_for_test[key], rp.coefs_for_test[key], device=torch.device("cpu")
    )
    compiled = torch.compile(proj, fullgraph=True, backend="aot_eager")
    y = torch.randn(4, HIDDEN)
    rp.fill_neutral(0.0)
    assert torch.equal(compiled(y), y)
    rp.fill_neutral(1.0)
    assert not torch.equal(compiled(y), y)
    torch.testing.assert_close(compiled(y), proj(y))


def test_parse_request_lambda(rp):
    assert rp.parse_request_lambda("refusal:1.0") == 1.0
    assert rp.parse_request_lambda("refusal:abc") is None
    assert rp.parse_request_lambda("other-salt") is None
    assert rp.parse_request_lambda(None) is None


def test_refusal_state_expands_per_request_lambdas(rp):
    from vllm.v1.worker.gpu import refusal_utils

    importlib.reload(refusal_utils)
    state = refusal_utils.RefusalState(4, 8, torch.device("cpu"))
    state.add_request(0, None)
    state.add_request(2, 1.0)
    # Batch order: slot 2 (3 tokens), slot 0 (2 tokens); 3 padding rows.
    state.fill(np.array([2, 0]), np.array([3, 2], dtype=np.int32), 5, 0.25)
    assert rp._tok_buf.tolist() == [1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25]

    state.remove_request(2)
    state.fill(np.array([2]), np.array([1], dtype=np.int32), 1, 0.0)
    assert rp._tok_buf[0].item() == 0.0


def test_refusal_state_unknown_layout_uses_global(rp):
    from vllm.v1.worker.gpu import refusal_utils

    importlib.reload(refusal_utils)
    state = refusal_utils.RefusalState(2, 8, torch.device("cpu"))
    state.add_request(0, 1.0)
    # Adaptive verification: real token count below the CPU upper bound.
    state.fill(np.array([0]), np.array([4], dtype=np.int32), 2, 0.0)
    assert torch.all(rp._tok_buf == 0.0)


def _chat(**extra):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}
    return ChatCompletionRequest.model_validate(body)


def test_chat_kwarg_maps_to_cache_salt(rp):
    req = _chat(chat_template_kwargs={"refusal_lambda": 1, "reasoning_effort": "xhigh"})
    assert req.cache_salt == "refusal:1.0"
    assert rp.parse_request_lambda(req.cache_salt) == 1.0
    # An explicit salt wins; no kwarg means no salt.
    assert _chat(cache_salt="abc", chat_template_kwargs={"refusal_lambda": 1}).cache_salt == "abc"
    assert _chat(chat_template_kwargs={"reasoning_effort": "low"}).cache_salt is None


@pytest.mark.parametrize("bad", ["x", 9, float("nan")])
def test_chat_kwarg_rejects_bad_lambda(rp, bad):
    with pytest.raises(Exception, match="refusal_lambda"):
        _chat(chat_template_kwargs={"refusal_lambda": bad})


def test_chat_kwarg_ignored_without_dial(monkeypatch):
    monkeypatch.delenv("VLLM_REFUSAL_DIRS", raising=False)
    assert _chat(chat_template_kwargs={"refusal_lambda": 1}).cache_salt is None
