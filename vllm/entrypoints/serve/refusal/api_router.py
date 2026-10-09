# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime control of the refusal projection's global lambda.

  POST /admin/refusal_lambda   {"lambda": 1.0}
  GET  /admin/refusal_lambda

Mounted only when VLLM_REFUSAL_DIRS is set. Per-request lambda
(cache_salt="refusal:<x>") does not need this endpoint.

The value goes to every worker through collective_rpc (with TP=2 the API
server and the workers are different processes) and all ranks must agree.

Requests without a salt share prefix-cache blocks computed at the global
lambda, so changing it resets the prefix cache. The reset only succeeds with
no request holding KV; if it fails the old lambda is restored and the call
returns 409.
"""

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger
from vllm.refusal_projection import is_enabled

logger = init_logger(__name__)

router = APIRouter()

# 0 = unmodified model, 1 = measured operating point. Above ~1.5 quality
# drops; at 2.5 refusals return and generation runs away.
LAMBDA_MIN = -1.0
LAMBDA_MAX = 4.0


class RefusalLambdaRequest(BaseModel):
    lambda_: float = Field(..., alias="lambda", ge=LAMBDA_MIN, le=LAMBDA_MAX)

    model_config = {"populate_by_name": True}


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


async def _set_all_ranks(raw_request: Request, value: float) -> list | None:
    """Per-rank results, or None if any rank disagrees."""
    results = await engine_client(raw_request).collective_rpc(
        "set_refusal_lambda", args=(value,)
    )
    applied = [r for r in results if r is not None]
    if not applied or any(abs(r - value) > 1e-9 for r in applied):
        logger.error("refusal lambda: ranks disagree after setting %s: %s", value, results)
        return None
    return applied


@router.post("/admin/refusal_lambda", dependencies=[Depends(validate_json_request)])
async def set_refusal_lambda(request: RefusalLambdaRequest, raw_request: Request):
    value = float(request.lambda_)
    previous = await engine_client(raw_request).collective_rpc("get_refusal_lambda")

    applied = await _set_all_ranks(raw_request, value)
    if applied is None:
        return JSONResponse(
            status_code=500,
            content={"error": "ranks did not apply the same lambda", "requested": value},
        )

    if not await engine_client(raw_request).reset_prefix_cache():
        old = float(previous[0])
        if await _set_all_ranks(raw_request, old) is None:
            return JSONResponse(
                status_code=500,
                content={"error": "prefix cache busy and lambda rollback failed"},
            )
        return JSONResponse(
            status_code=409,
            content={
                "error": "prefix cache in use; lambda left at previous value. "
                "Retry when idle or use cache_salt='refusal:<x>' per request.",
                "lambda": old,
            },
        )

    logger.info("refusal lambda set to %s on %d ranks", value, len(applied))
    return JSONResponse(content={"lambda": value, "ranks": len(applied)})


@router.get("/admin/refusal_lambda")
async def get_refusal_lambda(raw_request: Request):
    results = await engine_client(raw_request).collective_rpc("get_refusal_lambda")
    vals = [r for r in results if r is not None]
    consistent = bool(vals) and all(abs(r - vals[0]) <= 1e-9 for r in vals)
    return JSONResponse(
        content={
            "lambda": vals[0] if consistent else None,
            "consistent": consistent,
            "per_rank": results,
        }
    )


def attach_router(app: FastAPI):
    if not is_enabled():
        return
    logger.warning(
        "Refusal projection enabled: /admin/refusal_lambda mounted. It changes "
        "model behaviour at runtime; do not expose it publicly."
    )
    app.include_router(router)
