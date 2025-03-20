import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)
from .configuration_olmo import OlmoConfig


logger = logging.get_logger(__name__)


class OlmoLayerNorm(nn.Module):
    """LayerNorm but with no learnable weight or bias."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        return F.layer_norm(hidden_states.to(dtype=torch.float32), self.normalized_shape, None, None, eps=1e-5).to(
            orig_dtype
        )


class OlmoMLP(LlamaMLP):
    def __init__(self, config: OlmoConfig):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        del self.up_proj

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)))
        return down_proj


class OlmoAttention(LlamaAttention):
    pass


class OlmoDecoderLayer(LlamaDecoderLayer):
    pass


class OlmoModel(LlamaModel):
    pass


class OlmoForCausalLM(LlamaForCausalLM):
    pass
