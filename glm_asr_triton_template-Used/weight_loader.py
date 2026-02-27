"""
Weight Loader for GLM-ASR
Educational implementation using Triton for tile-based GPU programming

Loads pre-trained weights from safetensors format into the model.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch


def create_config_from_hf(hf_config):
    """Create GlmAsrConfig from HuggingFace config."""
    from model import GlmAsrConfig

    ac = hf_config.audio_config
    tc = hf_config.text_config

    return GlmAsrConfig(
        audio_hidden_size=ac.hidden_size,
        audio_num_heads=ac.num_attention_heads,
        audio_num_layers=ac.num_hidden_layers,
        audio_intermediate_size=ac.intermediate_size,
        audio_max_position_embeddings=getattr(ac, "max_position_embeddings", 1500),
        text_hidden_size=tc.hidden_size,
        text_num_heads=tc.num_attention_heads,
        text_num_kv_heads=tc.num_key_value_heads,
        text_num_layers=tc.num_hidden_layers,
        text_intermediate_size=tc.intermediate_size,
        text_vocab_size=tc.vocab_size,
        text_max_position_embeddings=tc.max_position_embeddings,
        text_rope_base=getattr(tc, "rope_theta", 10000.0),
        projector_hidden_size=4096,
        projector_pool_factor=4,
        pad_token_id=getattr(tc, "pad_token_id", 0)
        if getattr(tc, "pad_token_id", None) is not None
        else 0,
        bos_token_id=getattr(tc, "bos_token_id", 1)
        if getattr(tc, "bos_token_id", None) is not None
        else 1,
        eos_token_id=getattr(tc, "eos_token_id", 2),
    )


def load_linear_weight(triton_linear, hf_weight, hf_bias=None):
    """Load weight (and optional bias) into Triton Linear layer."""
    triton_linear.weight = hf_weight.detach().to(torch.float32).clone()
    if hf_bias is not None and triton_linear.has_bias:
        triton_linear.bias_param = hf_bias.detach().to(torch.float32).clone()


def load_conv1d_weight_from_hf(triton_conv, hf_weight, hf_bias=None):
    """Load weight into Triton Conv1d layer from HF format."""
    weight = hf_weight.detach().to(torch.float32)
    out_channels, in_channels, kernel_size = weight.shape
    triton_conv.weight = weight.reshape(out_channels, in_channels * kernel_size).clone()

    if triton_conv.use_triton and (
        triton_conv.col_size_padded != triton_conv.col_size
        or triton_conv.out_channels_padded != out_channels
    ):
        triton_conv.weight_padded = torch.zeros(
            (triton_conv.out_channels_padded, triton_conv.col_size_padded),
            dtype=torch.float32,
        )
        triton_conv.weight_padded[:out_channels, : triton_conv.col_size] = triton_conv.weight
    else:
        triton_conv.weight_padded = triton_conv.weight

    if hf_bias is not None and triton_conv.has_bias:
        triton_conv.bias = hf_bias.detach().to(torch.float32).clone()


def load_layernorm_weight_from_hf(triton_ln, hf_weight, hf_bias):
    """Load LayerNorm weights."""
    triton_ln.weight = hf_weight.detach().to(torch.float32).clone()
    triton_ln.bias = hf_bias.detach().to(torch.float32).clone()


def load_rmsnorm_weight_from_hf(triton_rms, hf_weight):
    """Load RMSNorm weight."""
    triton_rms.weight = hf_weight.detach().to(torch.float32).clone()


def load_embedding_weight_from_hf(triton_emb, hf_weight):
    """Load Embedding weight."""
    triton_emb.weight = hf_weight.detach().to(torch.float32).clone()


def load_weights_from_hf_model(model, hf_model) -> None:
    """
    Load weights from HuggingFace GLM-ASR model into Triton model.
    """
    hf_state = hf_model.state_dict()

    print("Loading audio encoder weights...")

    load_conv1d_weight_from_hf(
        model.audio_encoder.conv1,
        hf_state["audio_tower.conv1.weight"],
        hf_state["audio_tower.conv1.bias"],
    )
    load_conv1d_weight_from_hf(
        model.audio_encoder.conv2,
        hf_state["audio_tower.conv2.weight"],
        hf_state["audio_tower.conv2.bias"],
    )

    if "audio_tower.embed_positions.weight" in hf_state:
        model.audio_encoder.embed_positions = (
            hf_state["audio_tower.embed_positions.weight"]
            .detach()
            .to(torch.float32)
            .clone()
        )

    for i, layer in enumerate(model.audio_encoder.layers):
        prefix = f"audio_tower.layers.{i}"

        load_layernorm_weight_from_hf(
            layer.self_attn_layer_norm,
            hf_state[f"{prefix}.input_layernorm.weight"],
            hf_state[f"{prefix}.input_layernorm.bias"],
        )

        load_linear_weight(
            layer.q_proj,
            hf_state[f"{prefix}.self_attn.q_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.q_proj.bias"),
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f"{prefix}.self_attn.k_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.k_proj.bias"),
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f"{prefix}.self_attn.v_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.v_proj.bias"),
        )
        load_linear_weight(
            layer.out_proj,
            hf_state[f"{prefix}.self_attn.o_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.o_proj.bias"),
        )

        load_layernorm_weight_from_hf(
            layer.final_layer_norm,
            hf_state[f"{prefix}.post_attention_layernorm.weight"],
            hf_state[f"{prefix}.post_attention_layernorm.bias"],
        )

        load_linear_weight(
            layer.fc1,
            hf_state[f"{prefix}.mlp.fc1.weight"],
            hf_state[f"{prefix}.mlp.fc1.bias"],
        )
        load_linear_weight(
            layer.fc2,
            hf_state[f"{prefix}.mlp.fc2.weight"],
            hf_state[f"{prefix}.mlp.fc2.bias"],
        )

    load_layernorm_weight_from_hf(
        model.audio_encoder.layer_norm,
        hf_state["audio_tower.norm.weight"],
        hf_state["audio_tower.norm.bias"],
    )

    print("Loading multi-modal projector weights...")

    load_linear_weight(
        model.multi_modal_projector.linear_1,
        hf_state["multi_modal_projector.linear_1.weight"],
        hf_state["multi_modal_projector.linear_1.bias"],
    )
    load_linear_weight(
        model.multi_modal_projector.linear_2,
        hf_state["multi_modal_projector.linear_2.weight"],
        hf_state["multi_modal_projector.linear_2.bias"],
    )

    print("Loading text decoder weights...")

    load_embedding_weight_from_hf(
        model.text_decoder.embed_tokens,
        hf_state["language_model.model.embed_tokens.weight"],
    )

    for i, layer in enumerate(model.text_decoder.layers):
        prefix = f"language_model.model.layers.{i}"

        load_rmsnorm_weight_from_hf(
            layer.input_layernorm,
            hf_state[f"{prefix}.input_layernorm.weight"],
        )

        load_linear_weight(
            layer.q_proj,
            hf_state[f"{prefix}.self_attn.q_proj.weight"],
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f"{prefix}.self_attn.k_proj.weight"],
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f"{prefix}.self_attn.v_proj.weight"],
        )
        load_linear_weight(
            layer.o_proj,
            hf_state[f"{prefix}.self_attn.o_proj.weight"],
        )

        load_rmsnorm_weight_from_hf(
            layer.post_attention_layernorm,
            hf_state[f"{prefix}.post_attention_layernorm.weight"],
        )

        load_linear_weight(
            layer.mlp.gate_proj,
            hf_state[f"{prefix}.mlp.gate_proj.weight"],
        )
        load_linear_weight(
            layer.mlp.up_proj,
            hf_state[f"{prefix}.mlp.up_proj.weight"],
        )
        load_linear_weight(
            layer.mlp.down_proj,
            hf_state[f"{prefix}.mlp.down_proj.weight"],
        )

    load_rmsnorm_weight_from_hf(
        model.text_decoder.norm,
        hf_state["language_model.model.norm.weight"],
    )

    load_linear_weight(
        model.lm_head,
        hf_state["language_model.lm_head.weight"],
    )

    print("Weight loading complete!")


def _load_weights_from_safetensors(model, model_dir: str) -> None:
    """
    Memory-efficient weight loading: reads safetensors shards one tensor
    at a time instead of materializing the entire HF model in RAM.

    Peak RAM: ~Triton model size + 1 tensor, instead of 3x model size.
    """
    import gc
    import json
    from pathlib import Path
    from safetensors import safe_open

    model_path = Path(model_dir)

    # Determine which safetensors files to load
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    print(f"Loading weights from {len(shard_files)} shard(s)...")

    total_params = 0
    loaded = 0

    for shard_name in shard_files:
        shard_path = model_path / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                tensor = f.get_tensor(key).to(torch.float32)
                total_params += tensor.numel()
                _assign_weight(model, key, tensor)
                loaded += 1
                if loaded % 50 == 0:
                    print(f"  Loaded {loaded} params ({total_params / 1e6:.0f}M)...")
                del tensor
        gc.collect()

    print(f"Loaded {loaded} tensors, {total_params / 1e6:.1f}M parameters total")


def _assign_weight(model, key: str, tensor: torch.Tensor) -> None:
    """Assign a single weight tensor to the correct location in the Triton model."""
    # --- Audio encoder convolutions ---
    if key == "audio_tower.conv1.weight":
        load_conv1d_weight_from_hf(model.audio_encoder.conv1, tensor)
        return
    if key == "audio_tower.conv1.bias":
        model.audio_encoder.conv1.bias = tensor.clone()
        return
    if key == "audio_tower.conv2.weight":
        load_conv1d_weight_from_hf(model.audio_encoder.conv2, tensor)
        return
    if key == "audio_tower.conv2.bias":
        model.audio_encoder.conv2.bias = tensor.clone()
        return
    if key == "audio_tower.embed_positions.weight":
        model.audio_encoder.embed_positions = tensor.clone()
        return
    if key == "audio_tower.norm.weight":
        model.audio_encoder.layer_norm.weight = tensor.clone()
        return
    if key == "audio_tower.norm.bias":
        model.audio_encoder.layer_norm.bias = tensor.clone()
        return

    # Audio encoder layers
    if key.startswith("audio_tower.layers."):
        parts = key.split(".")
        layer_idx = int(parts[2])
        layer = model.audio_encoder.layers[layer_idx]
        rest = ".".join(parts[3:])

        if rest == "input_layernorm.weight":
            layer.self_attn_layer_norm.weight = tensor.clone()
        elif rest == "input_layernorm.bias":
            layer.self_attn_layer_norm.bias = tensor.clone()
        elif rest == "self_attn.q_proj.weight":
            layer.q_proj.weight = tensor.clone()
        elif rest == "self_attn.q_proj.bias":
            layer.q_proj.bias_param = tensor.clone()
        elif rest == "self_attn.k_proj.weight":
            layer.k_proj.weight = tensor.clone()
        elif rest == "self_attn.k_proj.bias":
            layer.k_proj.bias_param = tensor.clone()
        elif rest == "self_attn.v_proj.weight":
            layer.v_proj.weight = tensor.clone()
        elif rest == "self_attn.v_proj.bias":
            layer.v_proj.bias_param = tensor.clone()
        elif rest == "self_attn.o_proj.weight":
            layer.out_proj.weight = tensor.clone()
        elif rest == "self_attn.o_proj.bias":
            layer.out_proj.bias_param = tensor.clone()
        elif rest == "post_attention_layernorm.weight":
            layer.final_layer_norm.weight = tensor.clone()
        elif rest == "post_attention_layernorm.bias":
            layer.final_layer_norm.bias = tensor.clone()
        elif rest == "mlp.fc1.weight":
            layer.fc1.weight = tensor.clone()
        elif rest == "mlp.fc1.bias":
            layer.fc1.bias_param = tensor.clone()
        elif rest == "mlp.fc2.weight":
            layer.fc2.weight = tensor.clone()
        elif rest == "mlp.fc2.bias":
            layer.fc2.bias_param = tensor.clone()
        return

    # --- Multi-modal projector ---
    if key == "multi_modal_projector.linear_1.weight":
        model.multi_modal_projector.linear_1.weight = tensor.clone()
        return
    if key == "multi_modal_projector.linear_1.bias":
        model.multi_modal_projector.linear_1.bias_param = tensor.clone()
        return
    if key == "multi_modal_projector.linear_2.weight":
        model.multi_modal_projector.linear_2.weight = tensor.clone()
        return
    if key == "multi_modal_projector.linear_2.bias":
        model.multi_modal_projector.linear_2.bias_param = tensor.clone()
        return

    # --- Text decoder ---
    if key == "language_model.model.embed_tokens.weight":
        model.text_decoder.embed_tokens.weight = tensor.clone()
        return
    if key == "language_model.model.norm.weight":
        model.text_decoder.norm.weight = tensor.clone()
        return
    if key == "language_model.lm_head.weight":
        model.lm_head.weight = tensor.clone()
        return

    # Text decoder layers
    if key.startswith("language_model.model.layers."):
        parts = key.split(".")
        layer_idx = int(parts[3])
        layer = model.text_decoder.layers[layer_idx]
        rest = ".".join(parts[4:])

        if rest == "input_layernorm.weight":
            layer.input_layernorm.weight = tensor.clone()
        elif rest == "self_attn.q_proj.weight":
            layer.q_proj.weight = tensor.clone()
        elif rest == "self_attn.k_proj.weight":
            layer.k_proj.weight = tensor.clone()
        elif rest == "self_attn.v_proj.weight":
            layer.v_proj.weight = tensor.clone()
        elif rest == "self_attn.o_proj.weight":
            layer.o_proj.weight = tensor.clone()
        elif rest == "post_attention_layernorm.weight":
            layer.post_attention_layernorm.weight = tensor.clone()
        elif rest == "mlp.gate_proj.weight":
            layer.mlp.gate_proj.weight = tensor.clone()
        elif rest == "mlp.up_proj.weight":
            layer.mlp.up_proj.weight = tensor.clone()
        elif rest == "mlp.down_proj.weight":
            layer.mlp.down_proj.weight = tensor.clone()
        return


def load_model_from_hf(model_name: str = "zai-org/GLM-ASR-Nano-2512"):
    """
    Load GLM-ASR model from HuggingFace and create Triton version.

    Uses memory-efficient shard-by-shard loading from safetensors files
    instead of materializing the full HuggingFace model in RAM.
    """
    from transformers import AutoProcessor, AutoConfig
    from model import GlmAsrModel

    print(f"Loading HuggingFace model: {model_name}")

    hf_config = AutoConfig.from_pretrained(model_name)
    triton_config = create_config_from_hf(hf_config)

    print("Creating Triton model with config:")
    print(
        f"  Audio: hidden={triton_config.audio_hidden_size}, heads={triton_config.audio_num_heads}, layers={triton_config.audio_num_layers}"
    )
    print(
        f"  Text: hidden={triton_config.text_hidden_size}, heads={triton_config.text_num_heads}, kv_heads={triton_config.text_num_kv_heads}, layers={triton_config.text_num_layers}"
    )

    triton_model = GlmAsrModel(triton_config)

    # Memory-efficient: download safetensors to cache, load shard-by-shard
    print("Loading HuggingFace weights...")
    try:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
        _load_weights_from_safetensors(triton_model, model_dir)
    except (ImportError, Exception) as e:
        # Fallback: load full HF model (uses more RAM)
        print(f"Safetensors loading failed ({e}), falling back to full model load...")
        from transformers import GlmAsrForConditionalGeneration
        hf_model = GlmAsrForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        load_weights_from_hf_model(triton_model, hf_model)
        del hf_model
        import gc
        gc.collect()

    processor = AutoProcessor.from_pretrained(model_name)

    return triton_model, processor
