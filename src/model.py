"""
Model loading and residual-stream extraction with plain HuggingFace transformers.

Uses HF `output_hidden_states` rather than TransformerLens, which does not support
Qwen3.8 (qwen3_5 architecture, hybrid linear attention). hidden_states[l] is the
residual stream entering block l.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name, dtype=torch.bfloat16):
    """Load model + tokenizer. Accepts an alias from MODELS or a bare HF repo id."""
    repo = MODELS.get(name, name)
    device = get_device()
    print(f"Loading '{repo}' ({dtype}) on {device}...")

    # On CUDA let accelerate spread the model over all visible GPUs.
    # For Qwen3.8 (a multimodal checkpoint) this loads the text-only Qwen3_5ForCausalLM.
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device != "cuda":
        model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(repo)
    # Left padding keeps the last real token at position -1 for every row in a batch.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.get_text_config().num_hidden_layers
    print(f"Model loaded: {n_layers} layers")
    return model, tokenizer, n_layers


def format_chat(tokenizer, messages):
    """Apply the chat template, ending at the assistant generation prompt.

    Thinking mode is set explicitly: Qwen3.8's template defaults to it but Qwen3.5-0.8B's does not
    (it appends an empty think block), so without the flag the two models are read at different tokens.
    With it, every prompt ends in `<think>\\n`.
    """
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


@torch.no_grad()
def last_token_activations(model, tokenizer, prompts, n_layers):
    """
    Residual stream at the last token, for every layer, for a batch of formatted prompts.

    Returns:
        float32 tensor [batch, n_layers, d_model] on CPU
    """
    # The chat template already contains the special tokens. Padding to a multiple of 64
    # bounds the number of distinct input shapes (MPS caches one graph per shape).
    enc = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=64, add_special_tokens=False)
    enc = enc.to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    # hidden_states has n_layers + 1 entries; the last one is post final-norm, drop it.
    acts = torch.stack([h[:, -1, :] for h in out.hidden_states[:n_layers]], dim=1)
    return acts.float().cpu()
