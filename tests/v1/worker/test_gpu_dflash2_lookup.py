# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 lookup-augmented drafting kernels against plain-Python references."""

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
    fuse_draft,
    suffix_lookup,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ref_lookup(seq, k, nmin, nmax):
    """Longest suffix match, then most recent: (tokens, match_len, valid)."""
    t = len(seq) - 1
    best = (-1, -1)
    for e in range(nmin - 1, t):
        n = 0
        while n < nmax and e - n >= 0 and seq[e - n] == seq[t - n]:
            n += 1
        if n >= nmin and (n, e) > best:
            best = (n, e)
    if best[0] < 0:
        return [0] * k, 0, 0
    n, e = best
    valid = min(k, t - e)
    return list(seq[e + 1 : e + 1 + valid]) + [0] * (k - valid), n, valid


def _cases():
    g = torch.Generator().manual_seed(0)
    cases = [
        torch.randint(1, vocab, (n,), generator=g).tolist()
        for vocab, n in ((6, 200), (20, 500), (3, 120), (60, 900))
    ]
    cases.append([5, 6, 7, 8] * 30)  # match overlaps its own suffix
    body = torch.randint(1, 50, (300,), generator=g).tolist()
    cases.append(body + body[40:80])  # verbatim quote near the end
    cases.append(list(range(1, 200)))  # no match
    return cases


@pytest.mark.parametrize("k,nmin,nmax", [(7, 4, 32), (7, 6, 12), (15, 8, 64)])
def test_suffix_lookup_matches_reference(k, nmin, nmax):
    for seq in _cases():
        all_ids = torch.zeros((4, len(seq) + 8), dtype=torch.int32, device="cuda")
        all_ids[1, : len(seq)] = torch.tensor(seq, dtype=torch.int32)
        total = torch.zeros(4, dtype=torch.int32, device="cuda")
        total[1] = len(seq)
        idx_map = torch.ones(k, dtype=torch.int32, device="cuda")  # request state 1
        toks, mlen, valid = suffix_lookup(
            all_ids, total, idx_map, 1, k, idx_mapping_stride=k, nmax=nmax, nmin=nmin
        )
        want_toks, want_len, want_valid = ref_lookup(seq, k, nmin, nmax)
        assert int(mlen[0]) == want_len
        assert int(valid[0]) == want_valid
        assert toks[0].tolist()[:want_valid] == want_toks[:want_valid]


@pytest.mark.parametrize(
    "match_len,agree,expect",
    [(12, 0, True), (4, 0, False), (4, 2, True), (2, 7, False)],
)
def test_fuse_takes_strong_or_agreeing_matches(match_len, agree, expect):
    k = 7
    draft = torch.arange(100, 100 + k, dtype=torch.int64, device="cuda").view(1, k)
    look = torch.arange(200, 200 + k, dtype=torch.int32, device="cuda").view(1, k)
    look[0, :agree] = draft[0, :agree].to(torch.int32)
    use = torch.zeros((1, k), dtype=torch.int32, device="cuda")
    fuse_draft(
        draft,
        look,
        torch.tensor([match_len], dtype=torch.int32, device="cuda"),
        torch.tensor([k], dtype=torch.int32, device="cuda"),
        use,
        torch.ones(k, dtype=torch.int32, device="cuda"),
        torch.zeros((), dtype=torch.int64, device="cuda"),
        1,
        k,
        idx_mapping_stride=k,
        nmin=4,
        nstrong=8,
        agree_min=2,
    )
    assert bool(use.any()) == expect
    if expect:
        assert draft[0].tolist() == look[0].tolist()


def test_point_mass_rewrites_only_lookup_positions():
    k, top_k, vocab = 4, 3, 32
    draft_logits = torch.full((2, k, vocab), -float("inf"), device="cuda")
    cached = torch.zeros((2, k, top_k), dtype=torch.int64, device="cuda")
    # The selector cached candidates 1..3 for request state 1, step by step.
    for step in range(k):
        cached[1, step] = torch.tensor([1, 2, 3])
        draft_logits[1, step, 1:4] = torch.tensor([0.5, 0.3, 0.2])
    draft_tokens = torch.tensor([[1, 20, 2, 21]], dtype=torch.int64, device="cuda")
    use = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32, device="cuda")
    idx_map = torch.ones(k, dtype=torch.int32, device="cuda")
    _point_mass_draft_logits_kernel[(k,)](
        draft_logits,
        cached,
        draft_tokens,
        draft_tokens.stride(0),
        use,
        idx_map,
        k,
        draft_logits.stride(0),
        draft_logits.stride(1),
        num_steps=k,
        top_k=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        num_warps=1,
    )
    probs = draft_logits[1].softmax(-1)
    for step, tok in ((1, 20), (3, 21)):
        assert probs[step, tok] == 1.0
        assert cached[1, step].tolist() == [tok] * top_k
    for step in (0, 2):
        assert torch.isfinite(draft_logits[1, step, 1:4]).all()
        assert cached[1, step].tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "match_len,agree,want_tail_lookup",
    [(12, 0, True), (4, 0, False), (6, 3, True), (4, 1, False)],
)
def test_fuse_fills_tail_past_the_drafter_block(match_len, agree, want_tail_lookup):
    k, block = 7, 3
    draft = torch.arange(100, 100 + k, dtype=torch.int64, device="cuda").view(1, k)
    look = torch.arange(200, 200 + k, dtype=torch.int32, device="cuda").view(1, k)
    look[0, :agree] = draft[0, :agree].to(torch.int32)
    use = torch.zeros((1, k), dtype=torch.int32, device="cuda")
    fuse_draft(
        draft,
        look,
        torch.tensor([match_len], dtype=torch.int32, device="cuda"),
        torch.tensor([k], dtype=torch.int32, device="cuda"),
        use,
        torch.ones(k, dtype=torch.int32, device="cuda"),
        torch.zeros((), dtype=torch.int64, device="cuda"),
        1,
        k,
        draft_block=block,
        idx_mapping_stride=block,
        nmin=4,
        nstrong=8,
        agree_min=2,
    )
    # Tail positions always get a point mass: nothing else wrote their q.
    assert bool(use[0, block:].all())
    tail_from_lookup = bool((draft[0, block:] == look[0, block:].long()).all())
    assert tail_from_lookup == want_tail_lookup
