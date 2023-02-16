import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2LMHeadModel, AutoModelForCausalLM
from .utils import print_main
from pathlib import Path
from transformers.modeling_utils import no_init_weights

LANGUAGE_MODELS = [
    "gptj",
    "japanese-gpt2-medium"
]


def gptj_config():
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.num_layers = 28
    config.num_heads = 16
    config.hidden_size = 256 * config.num_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True
    config.gradient_checkpointing = True
    return config


def get_gptj(
    gradient_checkpointing: bool = True,
    from_pretrained=False,
) -> torch.nn.Module:
    """
    Loads GPTJ language model from HF
    """
    print_main("Loading GPTJ language model...")
    config = gptj_config()
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"
    if from_pretrained:
        raise NotImplemented("GPTJ pretrained not implemented")
    else:
        with no_init_weights():
            model = GPTNeoForCausalLM(config=config)
    return model

#CHANGED
#japanese-gpt2-small
def get_japanese_gpt2_small(
) -> torch.nn.Module:
    print_main("Loading japanese-gpt2-small")
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small")
    return model

#japanese-gpt2-medium
def get_japanese_gpt2_medium(
) -> torch.nn.Module:
    print_main("Loading japanese-gpt2-medium")
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
    return model