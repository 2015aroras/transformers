from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import layer_type_validation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import logging
from ...utils.generic import TransformersKwargs
from ..olmo3.modeling_olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3RMSNorm,
)
from ..olmoe.configuration_olmoe import OlmoeConfig
from ..olmoe.modeling_olmoe import (
    OlmoeForCausalLM,
    OlmoeModel,
    OlmoePreTrainedModel,
    OlmoeRotaryEmbedding,
    OlmoeSparseMoeBlock,
    load_balancing_loss_func,
)


logger = logging.get_logger(__name__)


class Olmoe2Config(OlmoeConfig):
    r"""
    This is the configuration class to store the configuration of a [`Olmoe2Model`]. It is used to instantiate an OLMoE2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [allenai/OLMoE2-1B-7B-0924](https://huggingface.co/allenai/OLMoE2-1B-7B-0924).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 100278):
            Vocabulary size of the OLMoE2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Olmoe2Model`]
        hidden_size (`int`, *optional*, defaults to 12888):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 100277):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 100257):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 128):
            Number of routed experts.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.01):
            The aux loss factor for the total loss.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize the topk probabilities.
        moe_intermediate_size (`int`, *optional*, defaults to 1024):
            Intermediate size of the routed expert.
        shared_mlp_intermediate_size (`int`, *optional*, defaults to 4096):
            Intermediate size of the shared MLP in MoE layers.
        mlp_only_layers (`list[int]`, *optional*, defaults to `[0]`):
            Indicate the layers in which to not use Olmoe2SparseMoeBlock
            The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
        sliding_window (`int`, *optional*, defaults to 4097):
            Size of the sliding window for sliding window attention.
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Defaults to full attention in each layer.

    ```python
    >>> from transformers import Olmoe2Model, Olmoe2Config

    >>> # Initializing a OLMoE2 7B style configuration
    >>> configuration = Olmoe2Config()

    >>> # Initializing a model from the OLMoE2 7B style configuration
    >>> model = Olmoe2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmoe2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=100278,
        hidden_size=12888,
        intermediate_size=4096,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=100277,
        bos_token_id=None,
        eos_token_id=100257,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        num_experts_per_tok=8,
        num_experts=128,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        moe_intermediate_size=1024,
        shared_mlp_intermediate_size=4096,
        mlp_only_layers=None,
        sliding_window=4097,
        layer_types=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            num_experts_per_tok=num_experts_per_tok,
            num_experts=num_experts,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            norm_topk_prob=norm_topk_prob,
            **kwargs,
        )

        self.moe_intermediate_size = moe_intermediate_size
        self.shared_mlp_intermediate_size = shared_mlp_intermediate_size
        self.mlp_only_layers = [0] if mlp_only_layers is None else mlp_only_layers

        self.sliding_window = sliding_window
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if i % 4 != 0 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        del self.clip_qkv


# Olmoe2 RMS norm reuses Olmo3 RMS norm, which handles low precision slightly differently than the original Olmoe.
class Olmoe2RMSNorm(Olmo3RMSNorm):
    pass


# Olmoe2 attention reuses Olmo3 attention, which provides sliding window attention, rather than Olmoe attention.
class Olmoe2Attention(Olmo3Attention):
    pass


# Olmoe2 uses MLPs with different intermediate sizes.
class Olmoe2MLP(nn.Module):
    def __init__(self, config: Olmoe2Config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return down_proj


# The OLMoE 2 sparse MoE block is identical that of OLMoE, except:
# - Sigmoid is applied to router logits rather than softmax.
class Olmoe2SparseMoeBlock(OlmoeSparseMoeBlock):
    def __init__(self, config: Olmoe2Config):
        super().__init__(config)
        self.experts = nn.ModuleList(
            [Olmoe2MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.sigmoid(router_logits.float())
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits


class Olmoe2DenseDecoderLayer(Olmo3DecoderLayer):
    pass


# The OLMoE 2 decoder layer is identical to the OLMo 3 decoder layer, except:
# - A hybrid MoE MLP is used instead of a regular MLP for most layers.
class Olmoe2DecoderLayer(Olmoe2DenseDecoderLayer):
    def __init__(self, config: Olmoe2Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.config = config

        if layer_idx in config.mlp_only_layers:
            self.mlp = Olmoe2MLP(config, intermediate_size=config.intermediate_size)
            self.block_sparse_moe = None
        else:
            self.mlp = Olmoe2MLP(config, intermediate_size=config.shared_mlp_intermediate_size)
            self.block_sparse_moe = Olmoe2SparseMoeBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        moe_hidden_states = None
        router_logits = None
        if self.block_sparse_moe:
            moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)

            output_router_logits = kwargs.get("output_router_logits", self.config.output_router_logits)
            if not output_router_logits:
                router_logits = None

        mlp_hidden_states = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        assert isinstance(mlp_hidden_states, torch.Tensor)

        hidden_states = mlp_hidden_states + moe_hidden_states if moe_hidden_states else mlp_hidden_states
        return hidden_states, router_logits


class Olmoe2RotaryEmbedding(OlmoeRotaryEmbedding):
    pass


class Olmoe2PreTrainedModel(OlmoePreTrainedModel):
    pass


# The OLMoE 2 model is identical to the OLMoE model, except:
# - Sliding window attention is used for 3 out of 4 layers.
class Olmoe2Model(OlmoeModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output_router_logits = kwargs.get("output_router_logits", self.config.output_router_logits)
        all_router_logits: Optional[tuple[torch.FloatTensor, ...]] = () if output_router_logits else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, router_logits = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.self_attn.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if all_router_logits is not None:
                assert router_logits is not None
                all_router_logits += (router_logits,)

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )


class Olmoe2ForCausalLM(OlmoeForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        assert isinstance(outputs, MoeModelOutputWithPast)

        hidden_states = outputs.last_hidden_state
        assert hidden_states is not None
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if kwargs.get("output_router_logits"):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "Olmoe2Config",
    "Olmoe2ForCausalLM",
    "Olmoe2Model",
    "Olmoe2PreTrainedModel",  # noqa: F822
]
