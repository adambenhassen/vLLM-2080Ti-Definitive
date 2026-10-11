<!-- markdownlint-disable MD001 MD041 -->
# vLLM on 2× RTX 2080 Ti: faster speculative decoding for Qwen3.8-27B

This branch (`sm75-fixes`) builds on
[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive), an SM75 (Turing)
build of vLLM. It adds the speculative-decoding, quantization and correctness work needed to serve
Swift-1.5 / Qwen3.8-27B on two RTX 2080 Ti 22 GB cards (NVLink, TP=2) as a fast agent backend.

## Results

Same box, same benchmark scripts, before and after this branch's changes. Swift-1.5 Qwen3.8-27B INT4
(AWQ + GPTQ W4A16), FP8 KV, MTP k=3, TP=2, 200 W per card.

| | Before | After | Gain |
|---|---:|---:|---:|
| 1 request, agent-style sampling (temp 1.0, thinking on) | 73 tok/s | 102–105 tok/s | **+40 %** |
| Prose, greedy | 74 tok/s | 101–104 tok/s | +38 % |
| Code, greedy | 106 tok/s | 138–144 tok/s | +30–35 % |
| 3 concurrent requests | ~165–171 tok/s | 203–216 tok/s | +20–30 % |
| 8 concurrent requests | 301 tok/s | 392–425 tok/s | +30–40 % |
| KV cache pool (INT4 / INT8 model) | 606K / 280K tokens | 669K / 337K tokens | +10 % / +20 % |

A single-user DFlash2 route reaches **387 tok/s** when the answer reproduces text from the prompt
(code edits, quoting), and 95–100 tok/s on ordinary output.

All of it is lossless (identical greedy output) except the INT4 `lm_head`, which shifts the output
distribution by KL ≈ 0.002 with tool calls still 100 % valid.

## What this branch adds

**Speculative decoding**

- **Vocab-truncated MTP draft head.** `VLLM_MTP_DRAFT_VOCAB=<ids.json>`: the MTP drafter scores a
  ~54K-token slice of `lm_head` instead of all 248K rows. Rows are sliced from the checkpoint's head
  at load time, dense or packed int4.
- **Probabilistic MTP drafting.** `SPECULATIVE_DRAFT_SAMPLE_METHOD=probabilistic` now reaches MTP
  routes in the launcher (it was DFlash-only); drafts are sampled at the request temperature.
- **DFlash2 lookup drafting.** `VLLM_DFLASH2_LOOKUP=1` fuses continuations of earlier occurrences of
  the current suffix (from the request's own tokens) into the drafted block, with a point-mass draft
  distribution so rejection sampling stays exact.
- **Adaptive long verify block.** With the lookup on and `SPECULATIVE_TOKENS` above the drafter's
  trained block, the target verifies up to 16 tokens only while a copy is running and the drafter's
  own 8 otherwise. Needs `NO_ASYNC_SCHEDULING=1` and capture sizes covering both widths.

**Correctness fixes**

- FlashInfer on SM75 silently produced wrong output for verify blocks above 8 tokens: FULL CUDA graphs
  replayed stale attention plans. The graph-stable wrappers now cover up to 15 draft tokens and the
  short DFlash2 width.
- Per-batch-size draft schedules (`SPECULATIVE_TOKENS_PER_BATCH_SIZE`) crashed at boot with MTP.
- Backport of vllm#50021 (bounds checks in GDN spec-decode state lookups), plus fixes for Mamba state
  with MTP + prefix caching, async spec-decode ordering, and quantized DFlash drafters and embeddings.

**Runtime refusal dial.** A rank-1 refusal projection with a per-request strength
(`VLLM_REFUSAL_DIRS`, `cache_salt: "refusal:<λ>"` or `chat_template_kwargs.refusal_lambda`).

**Observability**

- `VLLM_ENGINE_STALL_SENTINEL_S=<s>`: warns once per episode when the engine stops completing steps
  while requests are live.
- One `Request finished` log line per request: finish reason, token counts, elapsed time.
- `VLLM_CAPTURE_HIDDEN_STATES=<file>`: dumps final hidden states (the `lm_head` input) for calibration.
- `ENABLE_PER_REQUEST_METRICS` / `ENABLE_FORCE_INCLUDE_USAGE` launcher switches.

## Settings used for the results

| Setting | Agent routes (MTP, 8 slots) | Single-user route (DFlash2, 1 slot) |
|---|---|---|
| Model | Swift-1.5 INT4 with GPTQ int4 `lm_head` and int4 MTP | Swift-1.5 INT4 (BF16 head) + `Qwen3.8-27B-DFlash2-W4A16` |
| `SPECULATIVE_TOKENS` | 3 | 15 (drafter 7, adaptive 8/16 verify) |
| Draft settings | `SPECULATIVE_DRAFT_SAMPLE_METHOD=probabilistic`, `VLLM_MTP_DRAFT_VOCAB=<54K ids>` | `VLLM_DFLASH2_LOOKUP=1`, `NO_ASYNC_SCHEDULING=1` |
| Other | `VLLM_ENGINE_STALL_SENTINEL_S=60` | `COMPILATION_CONFIG_JSON` with capture sizes `[8,16]` |

The quantized-head and quantized-MTP checkpoints and the draft-vocab list were prepared outside this
repo.

## Build and run

Build and launch as upstream: `./build.sh`, then `./launcher.sh` (interactive) or
`NON_INTERACTIVE=1 ./launcher.sh` with a profile from `profiles/2x2080Ti/` plus the settings above.
See [docs/non-interactive-launch.md](docs/non-interactive-launch.md).

Do not run `update.sh` on a checkout of this branch: it replaces the tree with weicj's release source,
which drops every change above. The DFlash2 long-verify route then produces wrong output without
erroring.

## Credits and license

Apache-2.0, like [vLLM](https://github.com/vllm-project/vllm). Derived from
[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive): retain the upstream
license and attribution to `github.com/weicj` when redistributing. Ported work, Apache-2.0:
DFlash2 lookup drafting and the stall sentinel from [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen),
the refusal projection from
[pocharlies-org/qwen38-27b-rank1-refusal-projection](https://github.com/pocharlies-org/qwen38-27b-rank1-refusal-projection).
