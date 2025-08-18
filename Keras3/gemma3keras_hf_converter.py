import os
import argparse
import torch
import keras
from transformers import Gemma3Config, Gemma3ForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file
import numpy as np

# Set the Keras backend
os.environ["KERAS_BACKEND"] = "torch"

def create_weight_map(hf_config):
    """
    Creates a mapping from Keras weight names to Hugging Face weight names.
    """
    num_hidden_layers = hf_config.num_hidden_layers
    weight_map = {}

    # Top-level layers
    weight_map["token_embedding/embeddings"] = "model.embed_tokens.weight"
    weight_map["token_embedding/embeddings_lm_head"] = "lm_head.weight"  # Tied weight
    weight_map["final_normalization/scale"] = "model.norm.weight"

    # Transformer blocks
    for i in range(num_hidden_layers):
        keras_prefix = f"decoder_block_{i}"
        hf_prefix = f"model.layers.{i}"

        # Attention layers
        weight_map[f"{keras_prefix}/attention/query/kernel"] = f"{hf_prefix}.self_attn.q_proj.weight"
        weight_map[f"{keras_prefix}/attention/key/kernel"] = f"{hf_prefix}.self_attn.k_proj.weight"
        weight_map[f"{keras_prefix}/attention/value/kernel"] = f"{hf_prefix}.self_attn.v_proj.weight"
        weight_map[f"{keras_prefix}/attention/attention_output/kernel"] = f"{hf_prefix}.self_attn.o_proj.weight"

        # Attention normalization layers
        weight_map[f"{keras_prefix}/attention/query_norm/scale"] = f"{hf_prefix}.self_attn.q_norm.weight"
        weight_map[f"{keras_prefix}/attention/key_norm/scale"] = f"{hf_prefix}.self_attn.k_norm.weight"

        # Feed-forward (MLP) layers
        weight_map[f"{keras_prefix}/ffw_gating/kernel"] = f"{hf_prefix}.mlp.gate_proj.weight"
        weight_map[f"{keras_prefix}/ffw_gating_2/kernel"] = f"{hf_prefix}.mlp.up_proj.weight"
        weight_map[f"{keras_prefix}/ffw_linear/kernel"] = f"{hf_prefix}.mlp.down_proj.weight"

        # Block normalization layers
        weight_map[f"{keras_prefix}/pre_attention_norm/scale"] = f"{hf_prefix}.input_layernorm.weight"
        weight_map[f"{keras_prefix}/post_attention_norm/scale"] = f"{hf_prefix}.post_attention_layernorm.weight"
        weight_map[f"{keras_prefix}/pre_ffw_norm/scale"] = f"{hf_prefix}.pre_feedforward_layernorm.weight"
        weight_map[f"{keras_prefix}/post_ffw_norm/scale"] = f"{hf_prefix}.post_feedforward_layernorm.weight"

    return weight_map


def create_gemma3_1b_config(keras_weights_dict):
    """
    Create a proper Gemma 3 1B configuration based on the actual model architecture.
    """
    if not keras_weights_dict:
        raise ValueError("Keras weights dictionary is required to infer configuration.")

    # Infer parameters from Keras weights
    embed_weight = keras_weights_dict.get("token_embedding/embeddings")
    if embed_weight is None:
        raise ValueError("Could not find 'token_embedding/embeddings' in Keras weights.")
    vocab_size, hidden_size = embed_weight.shape
    print(f"Inferred from weights: vocab_size={vocab_size}, hidden_size={hidden_size}")

    # Get attention parameters
    q_weight = keras_weights_dict.get("decoder_block_0/attention/query/kernel")
    if q_weight is None:
        raise ValueError("Could not find 'decoder_block_0/attention/query/kernel'")
    num_attention_heads, _, head_dim = q_weight.shape
    print(f"Inferred from Q weights: num_attention_heads={num_attention_heads}, head_dim={head_dim}")

    k_weight = keras_weights_dict.get("decoder_block_0/attention/key/kernel")
    if k_weight is None:
        raise ValueError("Could not find 'decoder_block_0/attention/key/kernel'")
    num_key_value_heads, _, _ = k_weight.shape
    print(f"Inferred from K weights: num_key_value_heads={num_key_value_heads}")

    # Count layers
    num_hidden_layers = 0
    for name in keras_weights_dict.keys():
        if "decoder_block_" in name and "/pre_attention_norm/scale" in name:
            layer_num = int(name.split("decoder_block_")[1].split("/")[0])
            num_hidden_layers = max(num_hidden_layers, layer_num + 1)
    print(f"Inferred num_hidden_layers: {num_hidden_layers}")

    # Get intermediate size from MLP weights
    gate_weight = keras_weights_dict.get("decoder_block_0/ffw_gating/kernel")
    if gate_weight is not None:
        intermediate_size = gate_weight.shape[1]  # After transpose: (intermediate_size, hidden_size)
        print(f"Inferred intermediate_size: {intermediate_size}")
    else:
        intermediate_size = hidden_size * 4  # Fallback
        print(f"Using fallback intermediate_size: {intermediate_size}")

    print(">>> Creating Gemma 3 1B configuration...")
    
    # For text-only Gemma3 1B, try using global attention to avoid sliding window issues
    print(">>> Using global_attention for text-only model")
    
    hf_config = Gemma3Config(
        vocab_size=vocab_size,  # YOUR vocab size (262144)
        hidden_size=hidden_size,  # YOUR hidden size (1152) 
        num_hidden_layers=num_hidden_layers,  # YOUR layers (26)
        num_attention_heads=num_attention_heads,  # YOUR heads (4)
        num_key_value_heads=num_key_value_heads,  # YOUR KV heads (1)
        head_dim=head_dim,  # YOUR head dim (256)
        intermediate_size=intermediate_size,  # YOUR intermediate size (6912)
        
        # Standard Gemma3 parameters
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=8192,
        
        # Gemma 3 specific attributes
        query_pre_attn_scalar=head_dim,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=30.0,
        
        # REQUIRED: Add RoPE parameters (Gemma3 model init expects these)
        rope_local_base_freq=10000.0,  # Required by Gemma3TextModel init
        rope_local_attention_window=512,
        rope_short_base_freq=10000.0,
        rope_short_attention_window=512,
        
        # Use global_attention for all layers to avoid sliding window cache issues
        layer_types=["global_attention"] * num_hidden_layers,
        
        # Set sliding_window to a value (required by model)
        sliding_window=4096,  # Set to reasonable default to avoid errors
        
        # Model type and other attributes
        model_type="gemma3",
        tie_word_embeddings=True,
        use_cache=True,
    )
    
    # Try to load reference config for any missing attributes
    try:
        reference_config = AutoConfig.from_pretrained("google/gemma-3-1b-it")
        print(">>> Loaded reference config for additional attributes")
        
        # Copy non-dimensional attributes that might be needed, but DON'T overwrite layer_types
        for attr in ['initializer_range', 'pad_token_id', 'bos_token_id', 'eos_token_id']:
            if hasattr(reference_config, attr):
                setattr(hf_config, attr, getattr(reference_config, attr))
        
    except Exception as e:
        print(f">>> Could not load reference config ({e}), continuing with manual config")
    
    # Ensure layer_types is correct after any modifications
    hf_config.layer_types = ["global_attention"] * num_hidden_layers
    print(f">>> Final config - layer_types: {hf_config.layer_types[:3]}...")
    print(f">>> Final config - sliding_window: {hf_config.sliding_window}")
    
    return hf_config


def test_model_generation(model, tokenizer, test_prompt="Hello, how are you?"):
    """Test if the model can generate text properly"""
    print(f"\n>>> Testing model generation with prompt: '{test_prompt}'")
    
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        print(f"Input tokens: {inputs['input_ids']}")
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        # Set model to eval mode
        model.eval()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                num_return_sequences=1,
                use_cache=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: '{generated_text}'")
        
        if len(generated_text.strip()) > len(test_prompt.strip()):
            print("✓ Model generation appears to be working!")
            return True
        else:
            print("✗ Model generation failed - no new tokens generated")
            return False
            
    except Exception as e:
        print(f"✗ Model generation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def inspect_weight_statistics(tensor, name):
    """Print statistics about a weight tensor"""
    print(f"  {name}:")
    print(f"    Shape: {tensor.shape}")
    print(f"    Mean: {tensor.mean().item():.6f}")
    print(f"    Std: {tensor.std().item():.6f}")
    print(f"    Min: {tensor.min().item():.6f}")
    print(f"    Max: {tensor.max().item():.6f}")
    print(f"    Non-zero: {torch.count_nonzero(tensor).item()}/{tensor.numel()}")


def convert_keras_to_hf(keras_model_path, hf_output_path, model_size="1b", test_generation=False):
    """
    Converts a Keras Gemma 3 model to a Hugging Face Transformers model.
    
    Note: For text-only Gemma3 1B models, we use global_attention instead of 
    sliding window attention to avoid compatibility issues with converted weights.
    """
    print(">>> Loading Keras model...")
    print(">>> Note: Converting text-only Gemma3 model - using global_attention layers")
    keras_model = keras.models.load_model(keras_model_path, compile=False)
    print("Keras model loaded successfully.")

    # Extract weights to infer configuration
    keras_weights_dict = {w.path.replace(":0", ""): w.numpy() for w in keras_model.weights}
    
    print(f">>> Creating Gemma 3 {model_size} configuration...")
    hf_config = create_gemma3_1b_config(keras_weights_dict)
    
    # Print config for debugging
    print(f"Config - Hidden size: {hf_config.hidden_size}")
    print(f"Config - Attention heads: {hf_config.num_attention_heads}")
    print(f"Config - KV heads: {hf_config.num_key_value_heads}")
    print(f"Config - Head dim: {hf_config.head_dim}")
    print(f"Config - Layers: {hf_config.num_hidden_layers}")
    print(f"Config - Vocab size: {hf_config.vocab_size}")
    print(f"Config - Intermediate size: {hf_config.intermediate_size}")
    print(f"Config - Sliding window: {hf_config.sliding_window}")
    print(f"Config - Layer types: {getattr(hf_config, 'layer_types', 'Not set')[:3]}... (showing first 3)")
    
    # Initialize the model - let it use default attention implementation
    print(">>> Initializing Hugging Face model...")
    hf_model = Gemma3ForCausalLM(hf_config)
    print("Hugging Face model initialized.")

    weight_map = create_weight_map(hf_config)
    state_dict = {}

    print(">>> Starting weight conversion...")
    
    # Print some statistics about Keras weights
    print("\n>>> Keras model weight statistics (first 5):")
    for name, weight in list(keras_weights_dict.items())[:5]:
        tensor = torch.from_numpy(weight)
        inspect_weight_statistics(tensor, name)
    
    conversion_count = 0
    for keras_name, hf_name in weight_map.items():
        if keras_name not in keras_weights_dict:
            if "lm_head" in hf_name:
                # Use tied embeddings
                keras_name = "token_embedding/embeddings"
                print(f"  [INFO] Using tied embeddings for {hf_name}")
            else:
                print(f"  [WARNING] Weight not found in Keras model: {keras_name}")
                continue
        
        if keras_name not in keras_weights_dict:
            print(f"  [ERROR] Weight still not found: {keras_name}")
            continue
        
        weight = keras_weights_dict[keras_name]
        tensor = torch.from_numpy(weight).float()  # Ensure float32
        final_tensor = None

        # Handle different weight types
        if "attention/query/kernel" in keras_name:
            # Keras: (num_q_heads, hidden_size, head_dim) -> HF: (num_q_heads * head_dim, hidden_size)
            num_heads, hidden_size, head_dim = tensor.shape
            final_tensor = tensor.permute(0, 2, 1).reshape(num_heads * head_dim, hidden_size)
            print(f"  [*] Reshaped Q {keras_name} -> {final_tensor.shape}")
            
        elif "attention/key/kernel" in keras_name or "attention/value/kernel" in keras_name:
            # Keras: (num_kv_heads, hidden_size, head_dim) -> HF: (num_kv_heads * head_dim, hidden_size)
            num_kv_heads, hidden_size, head_dim = tensor.shape
            final_tensor = tensor.permute(0, 2, 1).reshape(num_kv_heads * head_dim, hidden_size)
            print(f"  [*] Reshaped KV {keras_name} -> {final_tensor.shape}")
            
        elif "attention/attention_output/kernel" in keras_name:
            # Keras: (num_heads, head_dim, hidden_size) -> HF: (hidden_size, num_heads * head_dim)
            if len(tensor.shape) == 3:
                num_heads, head_dim, hidden_size = tensor.shape
                final_tensor = tensor.permute(2, 0, 1).reshape(hidden_size, num_heads * head_dim)
            else:
                final_tensor = tensor.T
            print(f"  [*] Reshaped O {keras_name} -> {final_tensor.shape}")
            
        elif "kernel" in keras_name and ("ffw" in keras_name or "mlp" in keras_name):
            # MLP layers need transpose
            final_tensor = tensor.T.contiguous()
            print(f"  [*] Transposed MLP {keras_name} -> {final_tensor.shape}")
            
        else:
            # Embeddings, norms - usually no transformation needed
            final_tensor = tensor.contiguous()
            print(f"  [*] Converted {keras_name} -> {final_tensor.shape}")
        
        state_dict[hf_name] = final_tensor
        conversion_count += 1
    
    print(f"\n>>> Converted {conversion_count} weights total")
    
    # Load the converted weights into the model
    print(">>> Loading state dict into HuggingFace model...")
    try:
        missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")
        print(">>> Weight loading complete.")
    except RuntimeError as e:
        print(f">>> Error loading state dict: {e}")
        raise

    # Test generation before saving
    if test_generation:
        print(">>> Loading tokenizer for generation test...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-3-{model_size}-it")
            success = test_model_generation(hf_model, tokenizer)
            if not success:
                print(">>> WARNING: Model generation test failed!")
        except Exception as e:
            print(f">>> Could not test generation: {e}")

    # Save the model
    print(f">>> Saving Hugging Face model to {hf_output_path}...")
    
    # Ensure output directory exists
    os.makedirs(hf_output_path, exist_ok=True)
    
    # Force layer_types to be all global_attention before saving
    hf_config.layer_types = ["global_attention"] * hf_config.num_hidden_layers
    hf_config.sliding_window = 4096  # Ensure sliding_window is set
    
    # Save with proper configuration
    hf_model.save_pretrained(hf_output_path, safe_serialization=True)
    
    # Also save the config explicitly to ensure it's preserved
    hf_config.save_pretrained(hf_output_path)
    print(">>> Config saved explicitly.")
    print(f">>> Saved config has layer_types: {hf_config.layer_types[:3]}...")
    print(f">>> Saved config has sliding_window: {hf_config.sliding_window}")
    
    # Save the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-3-{model_size}-it")
        tokenizer.save_pretrained(hf_output_path)
        print(">>> Tokenizer saved as well.")
    except Exception as e:
        print(f">>> Could not save tokenizer: {e}")
    
    print("Hugging Face model saved successfully.")
    print("\nConversion finished!")
    
    # Final test with saved model
    # DOES NOT WORK YET
    if test_generation:
        print("\n>>> Testing saved model...")
        try:
            # Load config first to ensure it's used properly
            loaded_config = AutoConfig.from_pretrained(hf_output_path, local_files_only=True)
            print(f">>> Loaded config - vocab_size: {loaded_config.vocab_size}, hidden_size: {loaded_config.hidden_size}")
            print(f">>> Loaded config - layer_types: {loaded_config.layer_types[:3]}...")
            print(f">>> Loaded config - sliding_window: {loaded_config.sliding_window}")
            
            # Ensure layer_types are all global_attention (not sliding_attention)
            if any("sliding" in lt for lt in loaded_config.layer_types):
                print(">>> WARNING: Config has sliding attention layers, forcing all to global_attention")
                loaded_config.layer_types = ["global_attention"] * loaded_config.num_hidden_layers
            
            # Load with the saved configuration
            test_model = Gemma3ForCausalLM.from_pretrained(
                hf_output_path, 
                config=loaded_config,  # Explicitly use the saved config
                torch_dtype=torch.float32,
                local_files_only=True,
                trust_remote_code=False,
            )
            test_tokenizer = AutoTokenizer.from_pretrained(hf_output_path, local_files_only=True)
            success = test_model_generation(test_model, test_tokenizer, "The meaning of life is")
            if success:
                print("✓ Saved model generation test passed!")
            else:
                print("✗ Saved model generation test failed!")
        except Exception as e:
            print(f"✗ Could not test saved model: {e}")
            import traceback
            traceback.print_exc()
            print(">>> Try loading manually with proper configuration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned Keras 3 Gemma 3 model to Hugging Face Transformers format."
    )
    parser.add_argument(
        "keras_model_path",
        type=str,
        help="Path to the input Keras model file (.keras).",
    )
    parser.add_argument(
        "hf_output_path",
        type=str,
        help="Path to the output directory for the Hugging Face model.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="1b",
        help="The size of the Gemma 3 model (e.g., '1b').",
    )
    #parser.add_argument(
    #    "--no-test", ## DOES NOT WORK YET
    #    action="store_true",
    #    help="Skip generation testing.",
    #)

    args = parser.parse_args()
    convert_keras_to_hf(
        args.keras_model_path, 
        args.hf_output_path, 
        args.model_size,
        test_generation=False #not args.no_test does not work yet
    )
