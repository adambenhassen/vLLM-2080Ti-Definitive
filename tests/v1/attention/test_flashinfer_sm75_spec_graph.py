# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends import flashinfer
from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _config(*, capture_sizes=(8,), max_num_seqs=1, dcp=1, spec_tokens=7):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=spec_tokens),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                decode_mode=lambda: CUDAGraphMode.FULL,
            ),
            cudagraph_capture_sizes=list(capture_sizes),
            max_cudagraph_capture_size=max(capture_sizes),
        ),
        model_config=SimpleNamespace(dtype=torch.float16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        attention_config=SimpleNamespace(use_non_causal=False),
    )


def _attention_spec(dtype=torch.float16):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        dtype=dtype,
    )


def test_sm75_spec_prefill_graph_support_is_narrow(monkeypatch):
    monkeypatch.setattr(
        flashinfer.current_platform,
        "is_device_capability",
        lambda capability: capability == 75,
    )
    monkeypatch.setattr(flashinfer.envs, "VLLM_BATCH_INVARIANT", False)

    assert flashinfer._sm75_spec_prefill_graph_query_len(
        _config(), _attention_spec()
    ) == 8
    assert flashinfer._sm75_spec_prefill_graph_query_len(
        _config(capture_sizes=(4, 8), max_num_seqs=2, spec_tokens=3),
        _attention_spec(),
    ) == 4
    assert (
        flashinfer._sm75_spec_prefill_graph_query_len(
            _config(capture_sizes=(8,), max_num_seqs=2), _attention_spec()
        )
        is None
    )
    config = _config()
    config.speculative_config.num_speculative_tokens = 3
    config.compilation_config.cudagraph_capture_sizes = [8]
    config.compilation_config.max_cudagraph_capture_size = 8
    assert (
        flashinfer._sm75_spec_prefill_graph_query_len(config, _attention_spec())
        is None
    )
    assert (
        flashinfer._sm75_spec_prefill_graph_query_len(
            _config(capture_sizes=(1, 8)), _attention_spec()
        )
        is None
    )
    assert (
        flashinfer._sm75_spec_prefill_graph_query_len(
            _config(dcp=2), _attention_spec()
        )
        is None
    )


def test_sm75_spec_prefill_wrapper_reuses_graph_buffers(monkeypatch):
    created = []

    class FakeWrapper:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr(
        flashinfer, "BatchPrefillWithPagedKVCacheWrapper", FakeWrapper
    )
    monkeypatch.setattr(
        flashinfer, "get_flashinfer_layout_string", lambda _layout: "NHD"
    )

    builder = object.__new__(FlashInferMetadataBuilder)
    builder._sm75_spec_prefill_wrappers = {}
    builder._workspace_buffer = torch.empty(1)
    builder.cache_config = SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: object()
    )
    builder.model_config = SimpleNamespace(max_model_len=262144)
    builder.device = torch.device("cpu")
    builder.page_size = 16
    builder.num_qo_heads = 12
    builder.num_kv_heads = 2
    builder.head_dim = 256

    first = builder._get_sm75_spec_prefill_wrapper(1, 8, True)
    second = builder._get_sm75_spec_prefill_wrapper(1, 8, True)

    assert first is second
    assert len(created) == 1
    kwargs = created[0][1]
    assert kwargs["backend"] == "fa2"
    assert kwargs["use_cuda_graph"] is True
    assert kwargs["qo_indptr_buf"].shape == (2,)
    assert kwargs["paged_kv_indptr_buf"].shape == (2,)
    assert kwargs["paged_kv_indices_buf"].shape == (16384,)
    assert kwargs["paged_kv_last_page_len_buf"].shape == (1,)

    builder.set_workspace_buffer(torch.empty(2))
    third = builder._get_sm75_spec_prefill_wrapper(1, 8, True)
    assert third is not first
    assert len(created) == 2
    causal_wrapper = builder._get_sm75_spec_prefill_wrapper(1, 8, False)
    assert causal_wrapper is not third
    assert len(created) == 3

    concurrent_wrapper = builder._get_sm75_spec_prefill_wrapper(2, 4, True)
    assert concurrent_wrapper is not third
    assert len(created) == 4
    concurrent_kwargs = created[-1][1]
    assert concurrent_kwargs["qo_indptr_buf"].shape == (3,)
    assert concurrent_kwargs["paged_kv_indptr_buf"].shape == (3,)
    assert concurrent_kwargs["paged_kv_indices_buf"].shape == (32768,)
    assert concurrent_kwargs["paged_kv_last_page_len_buf"].shape == (2,)


def test_sm75_spec_prefill_graph_query_lens_cover_dynamic_schedule():
    """Every width a dynamic draft schedule verifies needs a stable wrapper."""
    config = _config(capture_sizes=(8, 16), max_num_seqs=2)
    assert flashinfer._sm75_spec_prefill_graph_query_lens(config, 8) == {8}
    config.speculative_config.num_speculative_tokens_per_batch_size = [
        [1, 1, 7],
        [2, 2, 4],
    ]
    assert flashinfer._sm75_spec_prefill_graph_query_lens(config, 8) == {8, 5}
