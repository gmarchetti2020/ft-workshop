"""
Set up Keras JAX backend
Import JAX and run a sanity check on TPU. TPUv6e-4 host offers 4 TPU cores with 32GB of memory each.
"""

import jax

print("TPU devices:\n",jax.devices(),"\n")

NUM_TPUS=jax.device_count()

import random
import os
import json
import sys
from utils import load_config

# Load configuration from config.conf into environment variables
load_config()

# Check for required credentials and configuration
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")
DATASET_NAME = os.environ.get("DATASET_NAME")

if not all([KAGGLE_USERNAME, KAGGLE_KEY, MODEL_NAME, DATASET_NAME]):
    print("❌ Error: KAGGLE_USERNAME, KAGGLE_KEY, MODEL_NAME, and DATASET_NAME must be set in config.conf.")
    sys.exit(1)

# The Keras 3 distribution API is only implemented for the JAX backend for now.
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation
# overhead
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

"""
A few configuration parameters
"""
# MODEL_NAME is now read from config.conf
# Deduce model size from name format: "gemma2[_instruct]_{2b,9b}_en"
#MODEL_SIZE = MODEL_NAME.split("_")[-2]
#assert MODEL_SIZE in ("2b", "7b")

# Dataset
DATASET_PATH = f"{DATASET_NAME}.jsonl"
#DATASET_URL = f"https://huggingface.co/datasets/databricks/{DATASET_NAME}/resolve/main/{DATASET_PATH}"

# Finetuned model
FINETUNED_MODEL_DIR = "/mnt/content/finetuned"
FINETUNED_KERAS_DIR = "/mnt/content/finetuned_keras"
FINETUNED_WEIGHTS_PATH = f"{FINETUNED_MODEL_DIR}/model.weights.h5"
FINETUNED_VOCAB_PATH = f"{FINETUNED_MODEL_DIR}/vocabulary.spm"

"""
Load model

To load the model with the weights and tensors distributed across TPUs, first create a new DeviceMesh.
DeviceMesh represents a collection of hardware devices configured for distributed computation
and was introduced in Keras 3 as part of the unified distribution API.
"""

import keras
keras.utils.set_random_seed(42)
# Run inferences at half precision
#keras.config.set_floatx("bfloat16")
# Train at mixed precision (enable for large batch sizes)
keras.mixed_precision.set_global_policy("mixed_bfloat16")
import keras_hub

# Create a device mesh with (1, NUM_TPUS) shape so that the weights are sharded across
# all TPUs.
device_mesh = keras.distribution.DeviceMesh(
            (1, NUM_TPUS),
            ["batch", "model"],
            devices=keras.distribution.list_devices())


"""
LayoutMap from the distribution API specifies how the weights and tensors should be sharded or replicated,
using the string keys, for example, token_embedding/embeddings below,
which are treated like regex to match tensor paths. Matched tensors are sharded with model dimensions (4 TPUs);
others will be fully replicated.
"""

model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh)

# Weights that match 'token_embedding/embeddings' will be sharded on 4  TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in the decoder
# attention layers
layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (model_dim, None, None)

layout_map["decoder_block.*attention_output.*kernel"] = (model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear.*kernel"] = (model_dim, None)

"""
ModelParallel allows you to shard model weights or activation tensors across all devices on the DeviceMesh.
In this case, some of the Gemma 7B model weights are sharded across 4 TPU chips
according to the layout_map defined above. Now load the model in the distributed way.
"""

model_parallel = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch")

keras.distribution.set_distribution(model_parallel)
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(f"{MODEL_NAME}")
gemma_lm.summary()

"""
Now verify that the model has been partitioned correctly. Let's take decoder_block_1 as an example.
"""

decoder_block_1 = gemma_lm.backbone.get_layer('decoder_block_1')
print(type(decoder_block_1))
for variable in decoder_block_1.weights:
      print(f'{variable.path:<58}  {str(variable.shape):<16}  {str(variable.value.sharding.spec)}')

"""
Inference before finetuning
"""

TEST_EXAMPLES = [
        "Lizzy has to ship 540 pounds of fish that are packed into 30-pound crates. If the shipping cost of each crate is $1.5, how much will Lizzy pay for the shipment?",
        "A school choir needs robes for each of its 30 singers. Currently, the school has only 12 robes so they decided to buy the rest. If each robe costs $2, how much will the school spend?",
        ]

# Prompt template for the training data and the finetuning tests
PROMPT_TEMPLATE = "Instruction:\n{instruction}\nResponse:\n{response}"

TEST_PROMPTS = [
        PROMPT_TEMPLATE.format(instruction=example, response="")
        for example in TEST_EXAMPLES
        ]

gemma_lm.compile(sampler="greedy")

print ("Before fine-tuning:\n")

for test_example in TEST_EXAMPLES:
        response = gemma_lm.generate(test_example, max_length=256)
        output = response[len(test_example) :]
        print(f"{test_example}\n{output!r}\n")

"""
Download and prepare dataset
"""

print("\nDowloading and preparing fine-tuning dataset...\n")
# dataset is in vertex ai format {"input_text": "...", "output_text": "..."}
#os.system(f"wget -nv -nc -O {DATASET_PATH} {DATASET_URL}")

def generate_training_data(training_ratio: int = 100) -> list[str]:
        assert 0 < training_ratio <= 100
        data = []
        with open(DATASET_PATH, 'r', encoding='utf-8') as file:
            for line in file:
                features = json.loads(line)
                # Format the data into the prompt template
                data.append(PROMPT_TEMPLATE.format(
                    instruction=features["input_text"],
                    response=features["output_text"]
                ))
        print("Shuffling training data...")
        random.shuffle(data)

        total_data_count = len(data)
        training_data_count = total_data_count * training_ratio // 100
        print(f"Training examples: {training_data_count}/{total_data_count}")
        return data[:training_data_count]

# Limit to % for test purposes

training_data = generate_training_data(training_ratio=100)


"""
Fine-tune with or without LoRA

LoRA is a fine-tuning technique which greatly reduces the number of trainable parameters for downstream tasks
by freezing the full weights of the model and inserting a smaller number of new trainable weights into the model.
Basically LoRA reparameterizes the larger full weight matrices by 2 smaller low-rank matrices AxB
to train and this technique makes training much faster and more memory-efficient.
"""

print ("\nFine-tuning...\n")

# Enable LoRA for the model and set the LoRA rank to 4.
#gemma_lm.backbone.enable_lora(rank=4) #DOES NOT SEEM TO WORK WITH SHARDING
#gemma_lm.summary()

# Limit the input sequence length to 128 to control memory usage.
gemma_lm.preprocessor.sequence_length = 256
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.001,)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras_hub.metrics.Perplexity(from_logits=True)],
        sampler="greedy")

BATCH_SIZE=2*NUM_TPUS



gemma_lm.fit(training_data, epochs=3, batch_size=BATCH_SIZE)

"""
Inference after fine-tuning
"""

print ("After fine-tuning:\n")
for prompt in TEST_PROMPTS:
        output = gemma_lm.generate(prompt, max_length=256)
        print(f"{output}\n{'- '*40}")

# Finetuned model

print ("\nSaving fine-tuned model weights...\n")

# Make sure the directory exists
os.system("mkdir -p "+FINETUNED_MODEL_DIR)
os.system("mkdir -p "+FINETUNED_KERAS_DIR)

gemma_lm.save_weights(FINETUNED_WEIGHTS_PATH, overwrite=True)

gemma_lm.preprocessor.tokenizer.save_assets(FINETUNED_MODEL_DIR)

print ("\nModel weights saved.\n")

print ("\nSaving fine-tuned model preset in keras format...\n")
gemma_lm.save_to_preset(FINETUNED_KERAS_DIR)
print ("\nDone.\n")
