# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request lambda of the runtime refusal projection (Model Runner V2).

Mirrors LoraState: the same idx_mapping / num_scheduled_tokens of the batch
expand per-request lambdas into the per-token buffer bound by every
RefusalProjection module.
"""

import numpy as np
import torch

from vllm import refusal_projection
from vllm.logger import init_logger

logger = init_logger(__name__)

# NaN = the request carried no lambda; use the server's global value.
NO_LAMBDA = np.nan


class RefusalState:
    def __init__(self, max_num_reqs: int, max_num_tokens: int, device: torch.device):
        self.lambdas = np.full(max_num_reqs, NO_LAMBDA, dtype=np.float32)
        # Created here, before the model is built and before CUDA graph capture.
        refusal_projection.ensure_token_buffer(max_num_tokens, device)
        self._warned_layout = False

    def add_request(self, req_index: int, refusal_lambda: float | None) -> None:
        self.lambdas[req_index] = NO_LAMBDA if refusal_lambda is None else refusal_lambda

    def remove_request(self, req_index: int) -> None:
        self.lambdas[req_index] = NO_LAMBDA

    def fill(
        self,
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_tokens: int,
        global_lambda: float,
    ) -> None:
        """Row i of the buffer <-> token i of the batch."""
        if int(num_scheduled_tokens.sum()) != num_tokens:
            # Adaptive verification compacts tokens on the GPU only, so the CPU
            # layout is an upper bound. Never guess: use the global lambda.
            if not self._warned_layout:
                self._warned_layout = True
                logger.warning(
                    "refusal projection: token layout is not known on the CPU "
                    "(adaptive verification); per-request lambda is ignored"
                )
            refusal_projection.fill_neutral(global_lambda)
            return
        lam = self.lambdas[idx_mapping]
        lam = np.where(np.isnan(lam), np.float32(global_lambda), lam)
        tok = np.repeat(lam, num_scheduled_tokens)
        refusal_projection.fill_tokens(torch.from_numpy(tok), global_lambda)

    def fill_neutral(self, global_lambda: float) -> None:
        refusal_projection.fill_neutral(global_lambda)
