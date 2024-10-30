import sys
from pathlib import Path
from typing import Optional

import olmo
import torch
from olmo.config import ModelConfig

from transformers import OlmoConfig, OlmoForCausalLM


def main(olmo_directory: Path, hf_directory: Path, hf_revision: Optional[str]):
    hf_revision = hf_revision or "main"

    in_vec = torch.randint(0, 50000, (2048, 2))

    print("Loading hf model")
    hf_config = OlmoConfig.from_pretrained(str(hf_directory), revision=hf_revision)
    hf_config._attn_implementation = "sdpa"
    hf_model = OlmoForCausalLM.from_pretrained(
        str(hf_directory), config=hf_config, revision=hf_revision, device_map="auto"
    ).to(device="cpu")

    print("Running hf model")
    hf_out = hf_model(in_vec)

    print("Loading OLMo model")
    model_config = ModelConfig.load(str(olmo_directory / "config.yaml"), key="model")
    model_config.init_device = "cpu"
    olmo_model = olmo.OLMo(model_config)

    print("Loading OLMo state")
    model_state = torch.load(str(olmo_directory / "model.pt"), map_location="cpu")
    model_state = {key.removeprefix("model."): val for key, val in model_state.items()}

    print("Loading state into OLMo")
    olmo_model.load_state_dict(model_state)

    print("Running OLMo model")
    olmo_out = olmo_model(in_vec)

    max_logit_diff = torch.max(torch.abs(hf_out.logits - olmo_out.logits))
    if max_logit_diff >= 1e-4:
        raise RuntimeError(max_logit_diff)

    print("Passed with max logit diff:", max_logit_diff)


if __name__ == "__main__":
    try:
        olmo_directory, hf_directory = sys.argv[1], sys.argv[2]
        hf_revision = sys.argv[3] if len(sys.argv) >= 4 else None
    except IndexError:
        raise RuntimeError(f"Usage: {sys.argv[0]} olmo_directory hf_directory [hf_revision]")

    main(Path(olmo_directory), Path(hf_directory), hf_revision)
