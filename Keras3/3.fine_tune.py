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

# Dataset
DATASET_PATH = f"{DATASET_NAME}.jsonl"

# Finetuned model
FINETUNED_MODEL_DIR = "/mnt/content/finetuned3"
FINETUNED_KERAS_DIR = "/mnt/content/finetuned_keras3"
FINETUNED_WEIGHTS_PATH = f"{FINETUNED_MODEL_DIR}/model.keras"
FINETUNED_VOCAB_PATH = f"{FINETUNED_MODEL_DIR}/vocabulary.spm"

EPOCHS=3
BATCH_PER_TPU=4
BATCH_SIZE=BATCH_PER_TPU*NUM_TPUS

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

data_parallel = keras.distribution.DataParallel(devices=jax.devices())
keras.distribution.set_distribution(data_parallel)

gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(f"{MODEL_NAME}")
gemma_lm.summary()


"""
Inference before finetuning
"""

TEST_EXAMPLES = [
        #"Lizzy has to ship 540 pounds of fish that are packed into 30-pound crates. If the shipping cost of each crate is $1.5, how much will Lizzy pay for the shipment?",
        #"A school choir needs robes for each of its 30 singers. Currently, the school has only 12 robes so they decided to buy the rest. If each robe costs $2, how much will the school spend?",
        "Peter has 25 apples to sell. He sells the first 10 for $1 each, the second lot of 10 for $0.75 each and the last 5 for $0.50 each. How much money does he make?",
        "Bea has $40. She wants to rent a bike for $4/hour. How many hours can she ride the bike?",
        ]

TEST_PROMPTS = [
        {"prompts": "user: "+example, "responses":""} for example in TEST_EXAMPLES
        ]

sampler = keras_hub.samplers.TopKSampler(k=5, seed=2)

gemma_lm.compile(sampler=sampler)

print ("Before fine-tuning:\n")

for test_prompt in TEST_PROMPTS:
        response = gemma_lm.generate(test_prompt, max_length=256)
        output = response[len(test_prompt["prompts"][0]):]
        print(test_prompt["prompts"][0]+f"\n{output}\n")

"""
Download and prepare dataset
"""

print("\nDowloading and preparing fine-tuning dataset...\n")

#os.system(f"wget -nv -nc -O {DATASET_PATH} {DATASET_URL}")

def generate_training_data():
        prompts=[]
        responses=[]
        with open(DATASET_PATH, 'r', encoding='utf-8') as file:
            for line in file:
                features = json.loads(line)
                # Format the data into the prompt template
                prompts.append("user: "+features["input_text"]),
                responses.append("model: "+features["output_text"])
        data={
                "prompts": prompts[:48000],
                "responses": responses[:48000]
            }
        return data

# Limit to 10% for test purposes

training_data = generate_training_data()


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

# Limit the input sequence length to 256 to control memory usage.
gemma_lm.preprocessor.sequence_length = 256
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.001,)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

# Compile and train
gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras_hub.metrics.Perplexity(from_logits=True)],
        sampler=sampler)

gemma_lm.fit(training_data, epochs=EPOCHS, batch_size=BATCH_SIZE, )

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

#gemma_lm.save_weights(FINETUNED_WEIGHTS_PATH, overwrite=True)

gemma_lm.save(FINETUNED_WEIGHTS_PATH, overwrite=True)

gemma_lm.preprocessor.tokenizer.save_assets(FINETUNED_MODEL_DIR)

print ("\nModel weights saved.\n")

print ("\nSaving fine-tuned model preset in keras format...\n")
gemma_lm.save_to_preset(FINETUNED_KERAS_DIR)
print ("\nDone.\n")
