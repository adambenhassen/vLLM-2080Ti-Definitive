# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3_5 MTP model."""

import json
import os
from collections.abc import Iterable

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import LocalArgmaxMixin, SupportsPP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = model_config.hf_text_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
        # missing from hf_quant_config.json exclude_modules. Force unquantized.
        # Ref: https://github.com/vllm-project/vllm/pull/38650
        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124
        bf16_mtp_draft = os.getenv("VLLM_QWOPUS_MTP_BF16_DRAFT") == "1"
        quant_name = quant_config.get_name() if quant_config else None
        fc_quant = (
            None
            if quant_name == "modelopt_fp4"
            or (bf16_mtp_draft and quant_name != "fp8")
            else quant_config
        )
        if bf16_mtp_draft:
            logger.info(
                "VLLM_QWOPUS_MTP_BF16_DRAFT=1: loading Qwen3.5 MTP "
                "fc/layer weights without the target quantization config."
            )
        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=fc_quant,
            prefix=f"{prefix}.fc",
        )

        # GPTQ: quantized checkpoints may exclude MTP from quantization via
        # quantization_config.dynamic with "-:pattern" entries. When detected,
        # disable quantization for MTP layers so they use unquantized params.
        original_quant = vllm_config.quant_config
        if bf16_mtp_draft and quant_name != "fp8":
            vllm_config.quant_config = None
        elif quant_config and quant_name not in ("modelopt_fp4",):
            hf_qc = getattr(model_config.hf_config, "quantization_config", None)
            if isinstance(hf_qc, dict):
                dynamic = hf_qc.get("dynamic", {})
                if any(k.startswith("-:") and "mtp" in k for k in dynamic):
                    vllm_config.quant_config = None
        try:
            self.layers = torch.nn.ModuleList(
                Qwen3_5DecoderLayer(
                    vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(self.num_mtp_layers)
            )
        finally:
            vllm_config.quant_config = original_quant
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen3NextSparseMoeBlock,
            "mlp",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Branch on the inputs, not the rank: the drafter is built entirely on
        # the last PP stage, where `is_first_rank` is False.
        if intermediate_tensors is None:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            assert hidden_states.shape[0] == positions.shape[-1]
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None
        hidden_states, residual = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[-1]]
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_5MTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        # Vocab-truncated draft head (VLLM_MTP_DRAFT_VOCAB): the drafter scores
        # only the listed token ids instead of the full lm_head. Rejection
        # sampling keeps the output exact; only acceptance can change.
        self.draft_lm_head = None
        draft_vocab = envs.VLLM_MTP_DRAFT_VOCAB
        if draft_vocab and get_pp_group().is_last_rank:
            with open(draft_vocab) as f:
                ids = torch.tensor(json.load(f), dtype=torch.long, device="cpu")
            if (
                ids.ndim != 1
                or ids.numel() == 0
                or ids.unique().numel() != ids.numel()
                or int(ids.min()) < 0
                or int(ids.max()) >= config.vocab_size
            ):
                raise ValueError(
                    f"VLLM_MTP_DRAFT_VOCAB={draft_vocab}: expected a list of "
                    f"distinct token ids in [0, {config.vocab_size})"
                )
            self._draft_vocab_ids_cpu = ids
            # Moved to the head's device in load_weights.
            self.register_buffer("draft_vocab_ids", ids.clone(), persistent=False)
            # The rows are sliced from the checkpoint's unquantized lm_head.
            self.draft_lm_head = ParallelLMHead(
                ids.numel(),
                config.hidden_size,
                quant_config=None,
                prefix=maybe_prefix(prefix, "draft_lm_head"),
            )
            self.draft_logits_processor = LogitsProcessor(ids.numel())
            logger.info(
                "MTP drafter uses a %d-token draft head (%s)", ids.numel(), draft_vocab
            )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Qwen3.5 MTP requires input_ids or inputs_embeds")
        spec_step_idx = int(kwargs.get("spec_step_idx", 0))
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        if self.draft_lm_head is not None:
            sub = self.draft_logits_processor(self.draft_lm_head, hidden_states)
            if sub is None:
                return None
            logits = sub.new_full((sub.shape[0], self.config.vocab_size), -float("inf"))
            return logits.index_copy_(1, self.draft_vocab_ids, sub)
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.draft_lm_head is not None:
            top = self.draft_logits_processor.get_top_tokens(
                self.draft_lm_head, hidden_states
            )
            return self.draft_vocab_ids[top]
        return super().get_top_tokens(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                    elif self.draft_lm_head is not None and name.endswith(
                        "lm_head.weight"
                    ):
                        ids = self._draft_vocab_ids_cpu.to(weight.device)
                        yield "draft_lm_head.weight", weight.index_select(0, ids)
                else:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(remap_weight_names(weights))
        if self.draft_lm_head is not None:
            if "draft_lm_head.weight" not in loaded:
                raise ValueError(
                    "VLLM_MTP_DRAFT_VOCAB needs an unquantized lm_head.weight in "
                    "the checkpoint to slice the draft head from"
                )
            self.draft_vocab_ids = self.draft_vocab_ids.to(
                self.draft_lm_head.weight.device
            )
        return loaded


class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.set_moe_parameters()
