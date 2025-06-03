import typing
from typing import Optional, Union

import torch
from olmo_core.nn.transformer import TransformerConfig

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, logging
from .configuration_olmo_core import OlmoCoreConfig


logger = logging.get_logger(__name__)


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring
class OlmoCorePreTrainedModel(PreTrainedModel):
    config_class = OlmoCoreConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = [
        "TransformerBlockBase",
        "TransformerBlock",
        "ReorderedNormTransformerBlock",
        "MoETransformerBlock",
        "MoEReorderedNormTransformerBlock",
        "MoEHybridTransformerBlockBase",
        "MoEHybridTransformerBlock",
        "MoEHybridReorderedNormTransformerBlock",
    ]
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_cache_class = False
    _supports_quantized_cache = False
    _supports_static_cache = False
    _supports_attention_backend = False


class OlmoCoreForCausalLM(OlmoCorePreTrainedModel, GenerationMixin):
    def __init__(self, config: OlmoCoreConfig):
        super().__init__(config)
        self.olmo_core_model = TransformerConfig.from_dict(config.olmo_core_config_dict).build()

    def get_input_embeddings(self):
        raise NotImplementedError

    def set_input_embeddings(self, value):
        raise NotImplementedError

    def get_output_embeddings(self):
        raise NotImplementedError

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError

    def set_decoder(self, decoder):
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    @can_return_tuple
    def _forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if past_key_values is not None:
            raise NotImplementedError("past_key_values")
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds")
        if output_attentions:
            raise NotImplementedError("output_attentions")
        if output_hidden_states:
            raise NotImplementedError("output_hidden_states")
        if use_cache:
            raise NotImplementedError("use_cache")

        logits = self.olmo_core_model(input_ids)

        return CausalLMOutputWithPast(logits=logits)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        if attention_mask is not None:
            raise NotImplementedError("attention_mask")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs: CausalLMOutputWithPast = self._forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        assert outputs.logits is not None
        logits = typing.cast(torch.FloatTensor, outputs.logits[:, slice_indices].float())

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs


__all__ = [
    "OlmoCoreForCausalLM",
    "OlmoCorePreTrainedModel",
]
