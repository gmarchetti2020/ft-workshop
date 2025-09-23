"""
Trains a Gemma 3.1B model with FFT and LoRA on a specified dataset.

This script handles the entire training pipeline, including data loading,
model configuration with LoRA, training, evaluation, and saving checkpoints.

Usage:
------
To run the training script, use the following command structure.
You must provide the required directory paths for checkpoints and profiling.

Example:
--------
```bash
python gemma3_1b_lora_trainer.py \
    --intermediate_ckpt_dir ./output/intermediate_ckpt/fft/v3/ \
    --ckpt_dir ./output/ckpts/medical/fft/v3/ \
    --profiling_dir ./output/profiling/ \
    --servable_ckpt_dir ./output/servable_ckpt/fft/v1/ \
    --batch_size 16 \
    --rank 16 \
    --max_steps 500 \
    --dataset_name "medalpaca/medical_meadow_medqa"
```

Arguments:
----------
    --batch_size (int, optional):
        Batch size for training.
        Default: 8

    --rank (int, optional):
        Rank for LoRA.
        Default: 8

    --alpha (float, optional):
        Alpha for LoRA.
        Default: 1.0

    --max_steps (int, optional):
        Maximum training steps.
        Default: 200

    --eval_every_n_steps (int, optional):
        Evaluate every N steps.
        Default: 20

    --num_epochs (int, optional):
        Number of training epochs.
        Default: 1

    --dataset_name (str, optional):
        The name of the dataset to use from the Hugging Face Hub.
        Default: "medalpaca/medical_meadow_medqa"

    --intermediate_ckpt_dir (str, required):
        Directory to save intermediate checkpoints during training.

    --ckpt_dir (str, required):
        Directory to save the final model checkpoint.

    --profiling_dir (str, required):
        Directory to save profiling output.

    --servable_ckpt_dir (str, required):
        Directory to save the servable model checkpoint.
"""


import json
import gc
import os
import time
import shutil
from flax import nnx
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
import qwix
from tunix.generate import sampler as sampler_lib
import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from huggingface_hub import snapshot_download
import argparse
import wandb
from safetensors.flax import save_file
from tunix.rl import common


os.environ["JAX_TRACEBACK_FILTERING"] = "off"

print("Welcome to Corrected Gemma3 1B LoRA Tuning")
print("Let's have some fun with Gemma3!")





def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Gemma3 1B IT Medical QA full finetuning Trainer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--rank", type=int, default=8, help="Rank for LoRA.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for LoRA.")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum training steps.")
    parser.add_argument("--eval_every_n_steps", type=int, default=20, help="Evaluate every N steps.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--dataset_name", type=str, default='medalpaca/medical_meadow_medqa', help="The name of the dataset to use.")
    parser.add_argument("--intermediate_ckpt_dir", type=str, required=True, default="./output/intermediate_ckpt/fft/v3/", help="Intermediate checkpoint directory.")
    parser.add_argument("--ckpt_dir", type=str,required=True, default="./output/ckpts/medical/fft/v3/", help="Checkpoint directory.")
    parser.add_argument("--profiling_dir", type=str, required=True, default="./output/profiling/", help="Profiling directory.")
    parser.add_argument("--servable_ckpt_dir", type=str, required=True, default="./output/servable_ckpt/fft/v1/", help="Servable checkpoint directory.")
    parser.add_argument("--test_prompts_file", type=str, default=None, help="Path to a file containing test prompts, one per line.")
    parser.add_argument("--local_file_path", type=str, default=None, help="Local path  to HF dataset file")
    return parser.parse_args()



def train_model(args):
    # Data - Optimized for instruction-tuned model
    BATCH_SIZE =args.batch_size # 8  # Very conservative for   model

    # Model
    print ("Checking Device shape .........")
    device_count = jax.device_count()
    print (f"Found ... {device_count} chips")
    print("Setting Mesh shape accordingly..")
    if device_count == 1:
        MESH = [(1, 1), ("fsdp", "tp")]
    elif device_count == 4:
      MESH = [(1, 4), ("fsdp", "tp")]
    else:
        MESH = [(2, 4), ("fsdp", "tp")]
    print(f"Using MESH: {MESH}")

    # LoRA - Very conservative for   model
    RANK =args.rank # 8  # Very small for   model
    ALPHA =args.alpha # 1.0  # Very small for   model

    # Train - Very conservative training for   model
    MAX_STEPS = args.max_steps # 2000  # Very few steps for   model
    EVAL_EVERY_N_STEPS = args.eval_every_n_steps # 20  # Frequent evaluation
    NUM_EPOCHS = args.num_epochs # 5  # Single epoch for   model

    print(f"LoRA Rank: {RANK}, Alpha: {ALPHA}")
    print(f"Training Steps: {MAX_STEPS}, Eval Every: {EVAL_EVERY_N_STEPS}")

    # Checkpoint saving
    INTERMEDIATE_CKPT_DIR = args.intermediate_ckpt_dir #"/home/shivajid/gemma3_1b/openmath/intckpt/"
    CKPT_DIR = args.ckpt_dir #"/home/shivajid/gemma3_1b/medqa/instruct/ckpt"
    PROFILING_DIR = args.profiling_dir #"/home/shivajid/gemma3_1b/openmath/profiling/"

    def chk_mkdir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    chk_mkdir(INTERMEDIATE_CKPT_DIR + "/state")
    chk_mkdir(CKPT_DIR)
    chk_mkdir(PROFILING_DIR)

    # Clean up existing checkpoints to avoid conflicts
    def cleanup_checkpoints():
        """Clean up existing checkpoints to avoid parameter conflicts."""
        if os.path.exists(CKPT_DIR):
            print(f"Removing existing checkpoint directory: {CKPT_DIR}")
            shutil.rmtree(CKPT_DIR)
        os.makedirs(CKPT_DIR, exist_ok=True)


    # Kaggle login
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        kagglehub.login()

    # Download Gemma3 1B   checkpoint
    kaggle_ckpt_path = kagglehub.model_download("google/gemma-3/flax/gemma3-1b/1")

    # Load Gemma3 1B   model
    model_config = gemma3_lib.Gemma3Config.gemma3_1b()


    def format_instruction_prompt(prompt):
        """Format prompt using the official Gemma3-1B-IT chat template."""
        return f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"

    # Create the model with proper weights from the   checkpoint
    def load_it_model_with_weights():
        """Load the   model with actual instruction-tuned weights."""
        # Load the actual weights from the   checkpoint
        it_checkpoint_path = os.path.join(kaggle_ckpt_path, "gemma3-1b")
        print(f"Loading   weights from: {it_checkpoint_path}")
        
        # Use the proper loading function from params_lib
        gemma3 = params_lib.create_model_from_checkpoint(
            checkpoint_path=it_checkpoint_path,
            model_config=model_config,
            mesh=None  # We'll shard it later
        )
        
        return gemma3

    # Load the model with actual   weights
    gemma3 = load_it_model_with_weights()


    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(gemma3)

    checkpoint_path = os.path.join(INTERMEDIATE_CKPT_DIR, "state")

    # Save checkpoint if it doesn't exist
    #if not os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)
    print(f"Saving checkpoint to: {checkpoint_path}")
    checkpointer.save(os.path.join(checkpoint_path), state)
    checkpointer.wait_until_finished()

    def get_base_model(ckpt_path):
        """Load the base Gemma3 1B   model from checkpoint."""
        model_config = gemma3_lib.Gemma3Config.gemma3_1b()
        mesh = jax.make_mesh(*MESH)
        abs_gemma3: nnx.Module = nnx.eval_shape(
            lambda: gemma3_lib.Gemma3(model_config, rngs=nnx.Rngs(params=0))
        )
        abs_state = nnx.state(abs_gemma3)
        abs_state = jax.tree.map(
            lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
            abs_state,
            nnx.get_named_sharding(abs_state, mesh),
        )
        checkpointer = ocp.StandardCheckpointer()
        restored_params = checkpointer.restore(ckpt_path, target=abs_state)

        graph_def, _ = nnx.split(abs_gemma3)
        gemma3 = nnx.merge(graph_def, restored_params)
        return gemma3, mesh, model_config

    # Base model
    gemma3, mesh, model_config = get_base_model(
        ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state")
    )

    # Load Gemma3 tokenizer
    print(f"Gemma3 tokenizer: {params_lib.GEMMA3_TOKENIZER}")
    TOKENIZER_PATH = os.path.join(kaggle_ckpt_path, 'tokenizer.model')
    gemma3_tokenizer = params_lib.create_tokenizer(TOKENIZER_PATH)

    # Test base model performance
    sampler = sampler_lib.Sampler(
        transformer=gemma3,
        tokenizer=gemma3_tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=512,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )


    # Test prompts with correct instruction format

    input_batch = [
        "Q:A pulmonary autopsy specimen from a 58-year-old woman who died of acute hypoxic respiratory failure was examined. She had recently undergone surgery for a fractured femur 3 months ago. Initial hospital course was uncomplicated, and she was discharged to a rehab facility in good health. Shortly after discharge home from rehab, she developed sudden shortness of breath and had cardiac arrest. Resuscitation was unsuccessful. On histological examination of lung tissue, fibrous connective tissue around the lumen of the pulmonary artery is observed. Which of the following is the most likely pathogenesis for the present findings??{'A': 'Thromboembolism', 'B': 'Pulmonary ischemia', 'C': 'Pulmonary hypertension', 'D': 'Pulmonary passive congestion', 'E': 'Pulmonary hemorrhage'},",
        "Q:A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms??{'A': 'Factor V Leiden', 'B': 'Hemophilia A', 'C': 'Lupus anticoagulant', 'D': 'Protein C deficiency', 'E': 'Von Willebrand disease'},",
        ]


    print("=== Base Model Performance ===")
    out_data = sampler(
        input_strings=input_batch,
        max_generation_steps=256,
    )

    for i, (input_string, out_string) in enumerate(zip(input_batch, out_data.text)):
        print(f"----------------------")
        print(f"----------------------")
        print(f"Test {i+1}:")
        print(f"Prompt:\n{input_string}")
        print(f"---****************-------")
        print(f"Output:\n\n\n{out_string}")
        print(f"---****************----")
        print(f"----------------------")


    def get_lora_model(base_model, mesh):
        """Apply LoRA to the base model."""
        lora_provider = qwix.LoraProvider(
            module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
            rank=RANK,
            alpha=ALPHA,
            # comment the two args below for LoRA (w/o quantisation).
            # weight_qtype="nf4",
            # tile_size=256,
        )

        # Create dummy model input for Gemma3 (similar to test_common.py)
        dummy_model_input = {
            'last_tokens': jnp.ones((2, 1), dtype=jnp.int32),
            'positions': jnp.ones((2, 1), dtype=jnp.int32),
            'cache': None,
            'attention_mask': jnp.ones((2, 1, 1), dtype=jnp.bool_),
        }

        lora_model = qwix.apply_lora_to_model(
            base_model, lora_provider, **dummy_model_input
        )

        with mesh:
            state = nnx.state(lora_model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(lora_model, sharded_state)

        return lora_model

    lora_gemma3 = get_lora_model(gemma3, mesh=mesh)


    # Load Datasets for SFT Training
    train_ds, validation_ds = data_lib.create_datasets(
        #dataset_name='mtnt/en-fr',
        dataset_name= args.dataset_name, #'medalpaca/medical_meadow_medqa' ,#'nvidia/OpenMathInstruct-1',
        global_batch_size=BATCH_SIZE,
        max_target_length=256,
        num_train_epochs=NUM_EPOCHS,
        tokenizer=gemma3_tokenizer,
        local_file_path=args.local_file_path,
        instruct_tuned=True,
    )



    def gen_model_input_fn(x: peft_trainer.TrainingInput):
        """Generate model inputs from training data."""
        pad_mask = x.input_tokens != gemma3_tokenizer.pad_id()
        positions = common.build_positions_from_mask(pad_mask)
        attention_mask = common.make_causal_attn_mask(pad_mask)
        return {
            'input_tokens': x.input_tokens,
            'input_mask': x.input_mask,
            'positions': positions,
            'attention_mask': attention_mask,
        }

    # PEFT Training
    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        checkpoint_root_directory=CKPT_DIR,
    )

    lora_trainer = peft_trainer.PeftTrainer(
        lora_gemma3, optax.adamw(1e-3), training_config
    ).with_gen_model_input_fn(gen_model_input_fn)

    with mesh:
        lora_trainer.train(train_ds, validation_ds)





    sampler = sampler_lib.Sampler(
        transformer=lora_gemma3,
        tokenizer=gemma3_tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=512,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    print("=== LoRA Fine-Tuned Model Performance ===")
    out_data = sampler(
        input_strings=input_batch,
        max_generation_steps=50,
    )

    for i, (input_string, out_string) in enumerate(zip(input_batch, out_data.text)):
        print(f"----------------------")
        print(f"----------------------")
        print(f"Test {i+1}:")
        print(f"Prompt:\n{input_string}")
        print(f"---****************-------")
        print(f"Output:\n\n\n{out_string}")
        print(f"---****************----")
        print(f"----------------------")

    ## generate
    print ("Merging checkpoint")
    print ("Loading checkpoint")

    trained_ckpt_path = os.path.join(CKPT_DIR, str(MAX_STEPS), "model_params")

    abs_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        nnx.state(lora_gemma3, nnx.LoRAParam),
    )

    print("restoring checkpoint")

    checkpointer = ocp.StandardCheckpointer()
    trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)


    print ("Merging checkpoint ....")
    nnx.update(
        lora_gemma3,
        jax.tree.map(
            lambda a, b: b,
            nnx.state(lora_gemma3, nnx.LoRAParam),
            trained_lora_params,
        ),
    )



    # ==============================================================================
    # ADD THIS CODE TO THE END OF YOUR SCRIPT TO SAVE IN SAFETENSORS FORMAT
    # ==============================================================================
    print("\nConverting and saving the fine-tuned model to .safetensors format...")

    # --- 1. Define the output directory ---
    # This is where your final .safetensors file will be saved.
    SERVABLE_CKPT_DIR = args.servable_ckpt_dir #"/home/shivajid/gemma3_1b/openmath/safetensors/"
    if os.path.exists(SERVABLE_CKPT_DIR):
        shutil.rmtree(SERVABLE_CKPT_DIR)
    os.makedirs(SERVABLE_CKPT_DIR, exist_ok=True)


    # --- 2. Define the corrected conversion function with updated LoRA weight names ---
    def model_state_to_hf_weights(state: nnx.State, rank: int, alpha: float) -> dict:
        """
        Converts a JAX/NNX Gemma state with LoRA adapters to a merged,
        Hugging Face compatible weight dictionary.

        This version uses the correct LoRA weight names found in your state file.
        """
        weights_dict = {}
        cpu = jax.devices("cpu")[0]
        scaling_factor = alpha / rank

        # Handle the embedding and final normalization weights
        vocab_size = 262144  # Standard Gemma vocab size
        weights_dict['model.embed_tokens.weight'] = jax.device_put(
            state.embedder.input_embedding.value, cpu
        )[:vocab_size, :]
        weights_dict['model.norm.weight'] = jax.device_put(state.final_norm.scale.value, cpu)

        # Gemma3 1B model dimensions
        embed_dim = 1152
        hidden_dim = 6 * 1152
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256

        print(f"Write state layers to file state_dump.txt")
        
        with open("state_dump.txt", "w", encoding="utf-8") as file:
            file.write(str(state))
        

        # Iterate through each layer to merge weights
        for idx, layer in state.layers.items():
            #if idx == 0:
             #print(f"layer: \n {layer} \n")
            weights_dict[f'model.layers.{idx}.input_layernorm.weight'] = jax.device_put(layer.pre_attention_norm.scale.value, cpu)
            weights_dict[f'model.layers.{idx}.post_attention_layernorm.weight'] = jax.device_put(layer.post_attention_norm.scale.value, cpu)
            # Corrected paths for q_norm and k_norm based on the provided state dump
            weights_dict[f'model.layers.{idx}.self_attn.q_norm.weight'] = jax.device_put(layer.attn._query_norm.scale.value, cpu)
            weights_dict[f'model.layers.{idx}.self_attn.k_norm.weight'] = jax.device_put(layer.attn._key_norm.scale.value, cpu)

            # Corrected paths for pre_feedforward_layernorm and post_feedforward_layernorm
            weights_dict[f'model.layers.{idx}.pre_feedforward_layernorm.weight'] =  jax.device_put(layer.pre_ffw_norm.scale.value, cpu)
            weights_dict[f'model.layers.{idx}.post_feedforward_layernorm.weight'] = jax.device_put(layer.post_ffw_norm.scale.value, cpu)




            # --- Attention Block ---
            attn = layer.attn

            # Q-Projection
            base_w_q = attn.q_einsum.w.value.transpose((0, 2, 1)).reshape((num_heads * head_dim, embed_dim))
            lora_A_q = attn.q_einsum.w_lora_a.value
            lora_B_q_raw = attn.q_einsum.w_lora_b.value
            #print(f"Lora shape lora_B_q_raw: {lora_B_q_raw.shape}")
            lora_B_q = lora_B_q_raw.reshape(rank, -1)

            #print(f"Lora shape Lora_B_q: {lora_B_q.shape}")
            #print(f"Lora shape Lora_A_q: {lora_A_q.shape}")
            #print(f"Base shape base_w_q: {base_w_q.shape}")
            delta_w_q = (lora_A_q @ lora_B_q) * scaling_factor
            weights_dict[f'model.layers.{idx}.self_attn.q_proj.weight'] = jax.device_put(base_w_q + delta_w_q.T, cpu)

            # K-Projection
            base_w_k = attn.kv_einsum.w.value[0].reshape(embed_dim, -1).T
            lora_A_k = attn.kv_einsum.w_lora_a.value
            lora_B_k_raw = attn.kv_einsum.w_lora_b.value
            #print(f"Lora shape base_w_k: {base_w_k.shape}")
            #print(f"Lora shape lora_A_k: {lora_A_k.shape}")
            #print(f"Lora shape lora_B_k_raw: {lora_B_k_raw.shape}")

            #lora_B_k = lora_B_k_raw.squeeze(0).T
            lora_B_k = lora_B_k_raw[:, 0, :, :].squeeze() if lora_B_k_raw.ndim == 4 else lora_B_k_raw
            #print(f"Corrected Lora shape lora_B_k: {lora_B_k.shape}")
            #print(f"Lora shape lora_B_k: {lora_B_k}")
            delta_w_k = (lora_A_k @ lora_B_k) * scaling_factor
            delta_w_k_T = delta_w_k.T
            weights_dict[f'model.layers.{idx}.self_attn.k_proj.weight'] = jax.device_put(base_w_k + delta_w_k_T, cpu)

            # V-Projection
            base_w_v = attn.kv_einsum.w.value[1].reshape(embed_dim, -1).T
            lora_A_v = attn.kv_einsum.w_lora_a.value
            lora_B_v_raw = attn.kv_einsum.w_lora_b.value
            #print(f"Lora shape base_w_v: {base_w_v.shape}")
            #print(f"Lora shape lora_A_v: {lora_A_v.shape}")
            #print(f"Lora shape lora_B_v_raw: {lora_B_v_raw.shape}")

            #lora_B_v = lora_B_v_raw.squeeze(0).T
            lora_B_v = lora_B_v_raw[:, 0, :, :].squeeze() if lora_B_v_raw.ndim == 4 else lora_B_v_raw
            #print(f"Corrected Lora shape lora_B_k: {lora_B_k.shape}")
            #print(f"Lora shape lora_B_v: {lora_B_v.shape}")
            delta_w_v = (lora_A_v @ lora_B_v) * scaling_factor
            delta_w_v_T = delta_w_v.T
            weights_dict[f'model.layers.{idx}.self_attn.v_proj.weight'] = jax.device_put(base_w_v + delta_w_v.T, cpu)

            # O-Projection (No LoRA applied in your config)
            
            #print(f"Attention vec shape {attn.attn_vec_einsum.w.value.shape}")
            base_w_o = attn.attn_vec_einsum.w.value.reshape(num_heads* head_dim, embed_dim).T
            weights_dict[f'model.layers.{idx}.self_attn.o_proj.weight'] = jax.device_put(base_w_o, cpu)

            # --- MLP Block ---
            mlp = layer.mlp

            # Gate Projection
            base_w_gate = mlp.gate_proj.kernel.value.T
            lora_A_gate = mlp.gate_proj.kernel_lora_a.value
            lora_B_gate_raw = mlp.gate_proj.kernel_lora_b.value
            #print(f"Lora shape base_w_gate: {base_w_gate.shape}")
            #print(f"Lora shape lora_A_gate: {lora_A_gate.shape}")
            #print(f"Lora shape lora_B_gate_raw: {lora_B_gate_raw.shape}")


            lora_B_gate = lora_B_gate_raw
            #print(f"Lora shape lora_B_gate: {lora_B_gate.shape}")
            delta_w_gate = (lora_A_gate @ lora_B_gate) * scaling_factor
            delta_w_gate_T = delta_w_gate.T
            weights_dict[f'model.layers.{idx}.mlp.gate_proj.weight'] = jax.device_put(base_w_gate + delta_w_gate_T, cpu)

            # Up Projection
            base_w_up = mlp.up_proj.kernel.value.T
            lora_A_up = mlp.up_proj.kernel_lora_a.value
            lora_B_up_raw = mlp.up_proj.kernel_lora_b.value
            #print(f"Lora shape base_w_up: {base_w_up.shape}")
            #print(f"Lora shape lora_A_up: {lora_A_up.shape}")
            #print(f"Lora shape lora_B_up_raw: {lora_B_up_raw.shape}")


            lora_B_up = lora_B_up_raw
            #print(f"Lora shape lora_B_up.T: {lora_B_up.shape}")
            delta_w_up = (lora_A_up @ lora_B_up) * scaling_factor
            delta_w_up_T = delta_w_up.T
            weights_dict[f'model.layers.{idx}.mlp.up_proj.weight'] = jax.device_put(base_w_up + delta_w_up_T, cpu)

            # Down Projection
            base_w_down = mlp.down_proj.kernel.value.T
            lora_A_down = mlp.down_proj.kernel_lora_a.value
            lora_B_down_raw = mlp.down_proj.kernel_lora_b.value
            #print(f"Lora shape base_w_down: {base_w_down.shape}")
            #print(f"Lora shape lora_A_down: {lora_A_down.shape}")
            #print(f"Lora shape lora_B_down_raw: {lora_B_down_raw.shape}")


            lora_B_down = lora_B_down_raw
            #print(f"Lora shape lora_B_down.T: {lora_B_down.shape}")
            delta_w_down = (lora_A_down @ lora_B_down) * scaling_factor
            delta_w_down_T = delta_w_down.T
            weights_dict[f'model.layers.{idx}.mlp.down_proj.weight'] = jax.device_put(base_w_down + delta_w_down_T, cpu)

        return weights_dict


    # --- 3. Execute the conversion and save the file ---

    # Extract the state (the weights) from the trained LoRA model object
    _, lora_state = nnx.split(lora_gemma3)


    # Perform the conversion on the CPU to be memory-safe
    with jax.default_device(jax.devices("cpu")[0]):
        hf_weights = model_state_to_hf_weights(lora_state,rank=RANK, alpha=ALPHA)

    print('Finished converting model weights to safetensors format')


    # vLLM wants the weight dictionary flattened
    def flatten_weight_dict(torch_params, prefix=""):
        flat_params = {}
        for key, value in torch_params.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat_params.update(flatten_weight_dict(value, new_key + "."))
            else:
                flat_params[new_key] = value
        return flat_params

    servable_weights = flatten_weight_dict(hf_weights)

    
    # Save the converted weights dictionary to a .safetensors file
    save_file(servable_weights, os.path.join(SERVABLE_CKPT_DIR, 'model.safetensors'))
    print(f" Model successfully saved to {os.path.join(SERVABLE_CKPT_DIR, 'model.safetensors')}")

    snapshot_download(repo_id="google/gemma-3-1b-it", allow_patterns="*.json", local_dir=SERVABLE_CKPT_DIR)

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
    print ("Training complete!")
    
