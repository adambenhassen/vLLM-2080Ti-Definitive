# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime rank-1 refusal projection for Qwen3.5/3.8 dense models.

Port of github.com/pocharlies-org/qwen38-27b-rank1-refusal-projection
(Apache-2.0, runtime/vllm-0.27.1/payload/refusal_projection.py).

Every module that writes to the residual stream gets

    y <- y - lam * coef_m * r_m * (r_m . y)

on its output, which equals editing the weight to (W - lam * coef_m * r r^T W)
without touching the checkpoint. lam=0 is the unmodified model; lam=1
reproduces the per-layer profile of the abliterated source checkpoint.

Enabled only when VLLM_REFUSAL_DIRS points at the direction file. Without it
no module is built and the forward has no extra branch.

lam sources:
  * global: VLLM_REFUSAL_LAMBDA_INIT at startup, POST /admin/refusal_lambda
    at runtime.
  * per request: cache_salt="refusal:<float>". The salt is already part of the
    prefix-cache block hash, so KV computed at different lam never mixes.

CUDA graphs and AOT compile: lam lives in device tensors registered as
non-persistent buffers and is only ever mutated in place. A Python float would
be baked into the captured graph and changing it would silently do nothing.
The per-token buffer is created by the V2 model runner before the model is
built, so each projection binds it at construction and the forward only slices
it.

The MTP drafter is never projected (the source checkpoint has no mtp.*
tensors). That costs draft acceptance on ablated requests, not correctness.
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_DIRS = "VLLM_REFUSAL_DIRS"
ENV_LAMBDA = "VLLM_REFUSAL_LAMBDA_INIT"
SALT_PREFIX = "refusal:"

# Key stem of the direction file. Runtime prefixes differ per wrapper
# (model.layers.N, language_model.model.layers.N), so only the layer index is
# taken from the runtime prefix.
_CKPT_STEM = "model.language_model.layers."
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

_lock = threading.Lock()
_dirs: dict[str, torch.Tensor] | None = None
_coefs: dict[str, float] | None = None
_consumed: set[str] = set()
_seen_prefixes: set[str] = set()
_lam_by_device: dict[torch.device, torch.Tensor] = {}
_lam_value: float = float(os.environ.get(ENV_LAMBDA, "0.0"))
_tok_buf: torch.Tensor | None = None
_warned_capacity: set[tuple[int, int]] = set()


def is_enabled() -> bool:
    return bool(os.environ.get(ENV_DIRS))


def _load_dirs(path: str) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Minimal safetensors reader: one F32 vector per module plus `__coefs__`,
    paired by the explicit `coef_order` metadata, never by implicit order."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        blob = fh.read()

    def _tensor(key: str) -> torch.Tensor:
        m = header[key]
        if m["dtype"] != "F32":
            raise ValueError(f"{path}: {key} is {m['dtype']}, expected F32")
        a, b = m["data_offsets"]
        return torch.frombuffer(bytearray(blob[a:b]), dtype=torch.float32).clone()

    if "__coefs__" not in header:
        raise ValueError(f"{path}: missing `__coefs__` (per-module coefficients)")
    dirs = {k: _tensor(k) for k in header if k not in ("__metadata__", "__coefs__")}
    order = json.loads(header.get("__metadata__", {})["coef_order"])
    cv = _tensor("__coefs__")
    if len(order) != len(cv) or set(order) != set(dirs):
        raise ValueError(
            f"{path}: coef_order ({len(order)}) does not match directions ({len(dirs)})"
        )
    return dirs, {m: float(cv[i]) for i, m in enumerate(order)}


def _ensure_loaded() -> tuple[dict[str, torch.Tensor] | None, dict[str, float] | None]:
    global _dirs, _coefs
    path = os.environ.get(ENV_DIRS)
    if not path:
        return None, None
    with _lock:
        if _dirs is None:
            _dirs, _coefs = _load_dirs(path)
            logger.info(
                "refusal projection: %d directions loaded from %s (coef %.4f..%.4f)",
                len(_dirs),
                path,
                min(_coefs.values()),
                max(_coefs.values()),
            )
    return _dirs, _coefs


def resolve_direction(prefix: str, sublayer: str) -> tuple[torch.Tensor, float] | None:
    """(r_hat, coef) for `<prefix>.<sublayer>`, or None if it is not projected."""
    dirs, coefs = _ensure_loaded()
    if dirs is None or coefs is None:
        return None
    _seen_prefixes.add(prefix)
    lowered = prefix.lower()
    # The MTP drafter reuses the decoder-layer class and also starts at index 0;
    # without this it would claim backbone layer 0's direction.
    if "mtp" in lowered or "draft" in lowered:
        return None
    m = _LAYER_RE.search(prefix)
    if m is None:
        return None
    key = f"{_CKPT_STEM}{int(m.group(1))}.{sublayer}"
    if key not in dirs:
        return None
    if key in _consumed:
        # A layer projected twice is 2*lam while the claim count still reads full.
        raise RuntimeError(f"refusal projection: {key} claimed twice")
    _consumed.add(key)
    return dirs[key], coefs[key]


def verify_all_consumed() -> None:
    """Fail closed: a half-ablated model raises no error and just behaves oddly."""
    dirs, _ = _ensure_loaded()
    if dirs is None:
        return
    orphan = sorted(set(dirs) - _consumed)
    if orphan:
        raise RuntimeError(
            f"refusal projection: {len(orphan)} directions not claimed by any "
            f"layer, e.g. {orphan[:3]}; runtime prefixes seen: "
            f"{sorted(_seen_prefixes)[:3]}. Refusing to serve a half-ablated model."
        )
    logger.info("refusal projection: %d/%d directions claimed", len(_consumed), len(dirs))


# ------------------------------------------------------------------ lambda state


def get_lambda_tensor(device: torch.device) -> torch.Tensor:
    """Global lam for `device`, shared by every layer. Construction time only."""
    with _lock:
        t = _lam_by_device.get(device)
        if t is None:
            # Created outside inference mode so it can be mutated in place later.
            with torch.inference_mode(False):
                t = torch.tensor(_lam_value, device=device, dtype=torch.float32)
            _lam_by_device[device] = t
        return t


def set_lambda(value: float) -> float:
    """Global lam, mutated in place on every device (safe after graph capture)."""
    global _lam_value
    with _lock, torch.inference_mode(False):
        for t in _lam_by_device.values():
            t.fill_(value)
        _lam_value = float(value)
        return _lam_value


def get_lambda() -> float:
    return _lam_value


def parse_request_lambda(cache_salt: str | None) -> float | None:
    """`refusal:<float>` -> lam; None if the salt is not ours or does not parse."""
    if not cache_salt or not cache_salt.startswith(SALT_PREFIX):
        return None
    try:
        return float(cache_salt[len(SALT_PREFIX) :])
    except ValueError:
        logger.warning("cache_salt %r does not parse as a lambda; ignored", cache_salt)
        return None


def ensure_token_buffer(max_num_tokens: int, device: torch.device) -> None:
    """Per-token lam buffer. Must exist before the model is built and captured."""
    global _tok_buf
    if not is_enabled():
        return
    with _lock:
        if _tok_buf is None:
            with torch.inference_mode(False):
                _tok_buf = torch.full(
                    (max_num_tokens,), _lam_value, dtype=torch.float32, device=device
                )
            logger.info(
                "refusal projection: per-token buffer ready (%d tokens, %s)",
                max_num_tokens,
                device,
            )


def fill_tokens(tok: torch.Tensor, global_lambda: float) -> None:
    """Real rows get `tok`, padding rows the global lam."""
    buf = _tok_buf
    if buf is None:
        return
    n = int(tok.shape[0])
    cap = int(buf.shape[0])
    if n > cap:
        # Never truncate: that would hand one request another request's lam.
        if (n, cap) not in _warned_capacity:
            _warned_capacity.add((n, cap))
            logger.warning(
                "refusal projection: batch needs %d rows, buffer has %d; "
                "using the global lambda for this batch",
                n,
                cap,
            )
        fill_neutral(global_lambda)
        return
    with torch.inference_mode(False):
        buf[:n].copy_(tok, non_blocking=True)
        if cap > n:
            buf[n:].fill_(float(global_lambda))


def fill_neutral(global_lambda: float) -> None:
    """Whole buffer to the global lam (dummy/profile runs)."""
    if _tok_buf is None:
        return
    with torch.inference_mode(False):
        _tok_buf.fill_(float(global_lambda))


# ------------------------------------------------------------------ the module


class RefusalProjection(nn.Module):
    """y <- y - lam * coef * r * (r . y); dot product in fp32."""

    def __init__(self, r_hat: torch.Tensor, coef: float, *, device: torch.device):
        super().__init__()
        self.register_buffer(
            "r_hat", r_hat.to(device=device, dtype=torch.float32), persistent=False
        )
        self.coef = float(coef)
        # Buffers, not plain attributes, so AOT-loaded graphs read the live
        # tensors instead of values serialized at compile time.
        self.register_buffer("lam", get_lambda_tensor(device), persistent=False)
        self.use_tok = _tok_buf is not None
        if self.use_tok:
            self.register_buffer("tok", _tok_buf, persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        r = self.r_hat
        proj = y.to(torch.float32) @ r
        lam = self.tok[: proj.shape[0]] if self.use_tok else self.lam
        return y - (lam * self.coef * proj).unsqueeze(-1).to(y.dtype) * r.to(y.dtype)
