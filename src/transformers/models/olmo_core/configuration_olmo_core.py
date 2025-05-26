from olmo_core.nn.transformer import TransformerConfig

from ...configuration_utils import PretrainedConfig


class OlmoCoreConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OlmoCoreModel`]. It is used to instantiate an OLMo Core
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [allenai/Olmo-Core-7B-1124-hf](https://huggingface.co/allenai/Olmo-Core-7B-1124-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50304):
            Vocabulary size of the Olmo Core model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OlmoCoreModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.

    ```python
    >>> from transformers import OlmoCoreForCausalLM, OlmoCoreConfig

    >>> # Initializing a Olmo Core 7B style configuration
    >>> configuration = OlmoCoreConfig()

    >>> # Initializing a model from the OlmoCore 7B style configuration
    >>> model = OlmoCoreForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo-core"
    has_no_defaults_at_init = False
    keys_to_ignore_at_inference = []
    base_model_tp_plan = {}
    base_model_pp_plan = {}

    def __init__(
        self,
        olmo_core_config_dict=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if olmo_core_config_dict is None:
            olmo_core_config_dict = TransformerConfig.olmo2_190M(vocab_size=4096).as_config_dict()

        self.olmo_core_config_dict = olmo_core_config_dict
        self.tie_word_embeddings = False
        self.use_cache = False

    @property
    def vocab_size(self) -> int:
        return self.olmo_core_config_dict["vocab_size"]

    @vocab_size.setter
    def vocab_size(self, vocab_size: int):
        self.olmo_core_config_dict["vocab_size"] = vocab_size

    @property
    def hidden_size(self) -> int:
        return self.olmo_core_config_dict["d_model"]

    @hidden_size.setter
    def hidden_size(self, hidden_size: int):
        self.olmo_core_config_dict["d_model"] = hidden_size

    @property
    def num_attention_heads(self) -> int:
        return self.olmo_core_config_dict["block"]["attention"]["n_heads"]

    @num_attention_heads.setter
    def num_attention_heads(self, num_attention_heads: int):
        self.olmo_core_config_dict["block"]["attention"]["n_heads"] = num_attention_heads

    @property
    def num_hidden_layers(self) -> int:
        return self.olmo_core_config_dict["n_layers"]

    @num_hidden_layers.setter
    def num_hidden_layers(self, num_hidden_layers: int):
        self.olmo_core_config_dict["n_layers"] = num_hidden_layers


__all__ = ["OlmoCoreConfig"]
