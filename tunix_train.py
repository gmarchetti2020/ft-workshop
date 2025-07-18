import functools
import gc
import os
from pprint import pprint
import re
import time

from flax import nnx
import grain
import humanize
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
from qwix import lora
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.rl.inference import inference_worker as inference
from tunix.rl.rollout import vanilla_rollout
from tunix.sft import metrics_logger

"""## Hyperparameters

Let's define the configuration we are going to use. Note that this is by no
means a "perfect" set of hyperparameters. To get good results, you might have
to train the model for longer.
"""

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 1.0

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
MESH = [(2, 4), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 2

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = 0.08
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = 0.2

# ====== Training ======
BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
NUM_BATCHES = 3738
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 100

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = f"{os.getcwd()}/intermediate_ckpt/"
CKPT_DIR = f"{os.getcwd()}/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

"""## Utility functions"""

def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)
  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")

"""## Data preprocessing

First, let's define some special tokens. We instruct the model to first reason
between the `<reasoning>` and `</reasoning>` tokens. After
reasoning, we expect it to provide the answer between the `<answer>` and
`</answer>` tokens.
"""

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""

"""We use OpenAI's GSM8K dataset. GSM8K comprises grade school math word problems."""

def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def get_dataset(data_dir, split="train") -> grain.MapDataset:
  # Download data
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )
  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=42)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": TEMPLATE.format(
                  system_prompt=SYSTEM_PROMPT,
                  question=x["question"].decode("utf-8"),
              ),
              # passed to reward functions
              "question": x["question"].decode("utf-8"),
              # passed to reward functions
              "answer": extract_hash_answer(x["answer"].decode("utf-8")),
          }
      )
  )
  return dataset

dataset = get_dataset(TRAIN_DATA_DIR, "train").batch(BATCH_SIZE)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)
  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test").batch(BATCH_SIZE)[
    :NUM_TEST_BATCHES
]

len(train_dataset), len(val_dataset) if val_dataset is not None else 0, len(
    test_dataset
)

"""Let's see how one batch of the dataset looks like!

"""

for ele in train_dataset[:1]:
  pprint(ele)

"""## Load the policy model and the reference model

The policy model is the model which is actually trained and whose weights are
updated. The reference model is the model with which we compute KL divergence.
This is to ensure that the policy updates are not huge and that it does not
deviate too much from the reference model.

Typically, the reference model is the base model, and the policy model is the
same base model, but with LoRA parameters. Only the LoRA parameters are updated.

Note: We perform full precision (fp32) training. You can, however, leverage
Qwix for QAT.

To load the model, you need to be on [Kaggle](https://www.kaggle.com/) and need
to have agreed to the Gemma license
[here](https://www.kaggle.com/models/google/gemma/flax/).
"""

# Log in
if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
  kagglehub.login()

kaggle_ckpt_path = kagglehub.model_download("google/gemma-2/flax/gemma2-2b-it")

# This is a workaround. The checkpoints on Kaggle don't work with NNX. So, we
# load the model, save the checkpoint locally, and then reload the model
# (sharded).
params = params_lib.load_and_format_params(
    os.path.join(kaggle_ckpt_path, "gemma2-2b-it")
)
gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(gemma)
checkpointer.save(os.path.join(, "state"), state)

# Wait for the ckpt to save successfully.
time.sleep(60)

# Delete the intermediate model to save memory.
del params
#del gemma
del state
gc.collect()

def get_ref_model(ckpt_path):
  mesh = jax.make_mesh(*MESH)
  model_config = gemma_lib.TransformerConfig.gemma2_2b()
  abs_gemma: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)
  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma, mesh, model_config

def get_lora_model(base_model, mesh):
  lora_provider = lora.LoraProvider(
      module_path=(
          ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
          ".*attn_vec_einsum"
      ),
      rank=RANK,
      alpha=ALPHA,
  )
  model_input = base_model.get_model_input()
  lora_model = lora.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )
  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)
  return lora_model

# Reference model
gemma, mesh, model_config = get_ref_model(
    ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state")
)
nnx.display(gemma)

# Policy model
lora_gemma = get_lora_model(gemma, mesh=mesh)
nnx.display(lora_gemma)

"""## Define reward functions

We define four reward functions:

- reward if the format of the output exactly matches the instruction given in
`TEMPLATE`;
- reward if the format of the output approximately matches the instruction given
in `TEMPLATE`;
- reward if the answer is correct/partially correct;
- Sometimes, the text between `<answer>`, `</answer>` might not be one
  number. So, extract the number, and reward the model if the answer is correct.

The reward functions are inspired from
[here](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb).

First off, let's define a RegEx for checking whether the format matches.
"""

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_format.search(
    f"{reasoning_start}Let me"
    f" think!{reasoning_end}{solution_start}2{solution_end}",
)

"""Give the model a reward of 3 points if the format matches exactly."""

def match_format_exactly(prompts, completions, **kargs):
  scores = []
  for completion in completions:
    score = 0
    response = completion
    # Match if format is seen exactly!
    if match_format.search(response) is not None:
      score += 3.0
    scores.append(score)
  return scores

"""We also reward the model if the format of the output matches partially."""

def match_format_approximately(prompts, completions, **kargs):
  scores = []
  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores

"""Reward the model if the answer is correct. A reward is also given if the answer
does not match exactly, i.e., based on how close the answer is to the correct
value.
"""

def check_answer(prompts, completions, answer, **kargs):
  responses = completions
  extracted_responses = [
      guess.group(1) if (guess := match_format.search(r)) is not None else None
      for r in responses
  ]
  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += 3.0
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += 0.5
        elif ratio >= 0.8 and ratio <= 1.2:
          score += 0.25
        else:
          score -= 1.0  # Penalize wrong answers
      except:
        score -= 0.5  # Penalize
    scores.append(score)
  return scores

"""Sometimes, the text between `<answer>` and `</answer>` might not be one
number; it can be a sentence. So, we extract the number and compare the answer.
"""

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")

def check_numbers(prompts, completions, answer, **kargs):
  question = kargs["question"]
  responses = completions
  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]
  scores = []
  print("START ============================")
  print(f"Question: {question[0]}")
  print(f"Answer: {answer[0]}")
  print(f"Response: {responses[0]}")
  print(f"Extracted: {extracted_responses[0]}")
  print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except:
      scores.append(0)
      continue
  return scores

"""## Evaluate


Before we train the model, let's evaluate the model on the test set so we can
see the improvement post training.

We evaluate it in two ways:

**Quantitative**

* **Answer Accuracy**: percentage of samples for which the model predicts the
correct final numerical answer  
* **Answer (Partial) Accuracy**: percentage of samples for which the model
predicts a final numerical answer such that the \`model answer / answer\`
ratio lies between 0.9 and 1.1.  
* **Format Accuracy**: percentage of samples for which the model outputs the
correct format, i.e., reasoning between the reasoning special tokens, and the
final answer between the \`\<start\_answer\>\`, \`\<end\_answer\>\` tokens.

**Qualitative**

We'll also print outputs for a few given questions so that we can compare the generated output later.

"""

def generate(
    question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None
):
  """Given prompt, generates text."""
  if isinstance(question, str):
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
        ),
    ]
  else:
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=q,
        )
        for q in question
    ]
  out_data = sampler(
      input_strings=input_batch,
      total_generation_steps=768,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=jax.random.PRNGKey(seed) if seed is not None else None,
  )
  output = out_data.text
  if isinstance(question, str):
    return output[0]
  return output

def evaluate(
    dataset,
    sampler,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """Computes accuracy and percentage of outputs matching the format."""
  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0
  for batch in tqdm(dataset):
    answers = batch["answer"]
    questions = batch["question"]
    multiple_call_responses = [[] for _ in range(len(questions))]
    for p in range(num_passes):
      responses = generate(
          questions, sampler, temperature, top_k, top_p, seed=p
      )
      for idx, response in enumerate(responses):
        multiple_call_responses[idx].append(response)
    for question, multiple_call_response, answer in zip(
        questions, multiple_call_responses, answers
    ):
      # check answer
      corr_ctr_per_question = 0
      partially_corr_per_question = 0
      corr_format_per_question = 0
      for response in multiple_call_response:
        extracted_response = (
            guess.group(1)
            if (guess := match_numbers.search(response)) is not None
            else "-1000000"
        )
        try:
          if float(extracted_response.strip()) == float(answer.strip()):
            corr_ctr_per_question += 1
          ratio = float(extracted_response.strip()) / float(answer.strip())
          if ratio >= 0.9 and ratio <= 1.1:
            partially_corr_per_question += 1
        except:
          print("SKIPPED")
        # check format
        if match_format.search(response) is not None:
          corr_format_per_question += 1
        if (
            corr_ctr_per_question > 0
            and partially_corr_per_question > 0
            and corr_format_per_question > 0
        ):
          break
      if corr_ctr_per_question > 0:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      if partially_corr_per_question > 0:
        partially_corr += 1
      if corr_format_per_question > 0:
        corr_format += 1
      total += 1
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )
  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )
  if make_lst:
    return to_return, response_lst
  return to_return

gemma_tokenizer = data_lib.GemmaTokenizer()
sampler = sampler_lib.Sampler(
    transformer=lora_gemma,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)

# for eval_example in QUALITATIVE_EVAL_EXAMPLES:
#   question = eval_example["question"]
#   answer = eval_example["answer"]
#   response = generate(
#       question,
#       sampler,
#       temperature=INFERENCE_TEMPERATURE,
#       top_k=INFERENCE_TOP_K,
#       top_p=INFERENCE_TOP_P,
#   )

#   print(f"Question:\n{question}")
#   print(f"Answer:\n{answer}")
#   print(f"Response:\n{response}")
#   print("===============")

"""## Train

Let's set up all the configs first - checkpointing, metric logging and training.
We then train the model.
"""

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/grpo", flush_every_n_steps=20
)

# Commented out IPython magic to ensure Python compatibility.
# Logs
# %load_ext tensorboard
# %tensorboard --logdir /content/tmp/tensorboard/grpo --port=0

# Training config
training_config = GrpoConfig(
    max_prompt_length=MAX_PROMPT_LENGTH,
    total_generation_steps=TOTAL_GENERATION_STEPS,
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    # metrics logging
    metrics_logging_options=metrics_logging_options,
    # checkpoint saving
    checkpoint_root_directory=CKPT_DIR,
    checkpointing_options=checkpointing_options,
)

# Rollout worker
gemma_tokenizer = data_lib.GemmaTokenizer()
rollout_worker = vanilla_rollout.VanillaRollout(
    model=lora_gemma,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)
inference_worker = inference.InferenceWorker(
    models=[
        inference.ModelContainer(
            model=gemma,
            role=inference.ModelRole.REFERENCE,  # use the base model as reference
        )
    ]
)

# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )

# GRPO Trainer
grpo_trainer = GrpoLearner(
    model=lora_gemma,
    inference_worker=inference_worker,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    rollout_worker=rollout_worker,
    optimizer=optimizer,
    training_config=training_config,
    trainer_mesh=mesh,
    rollout_worker_mesh=mesh,
)

with mesh:
  grpo_trainer.train(dataset)

"""## Evaluate

Let's evaluate our model!
"""

# Load checkpoint first.

trained_ckpt_path = os.path.join(CKPT_DIR, str(MAX_STEPS), "model_params")

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_gemma, nnx.LoRAParam),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    lora_gemma,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_gemma, nnx.LoRAParam),
        trained_lora_params,
    ),
)

gemma_tokenizer = data_lib.GemmaTokenizer()
sampler = sampler_lib.Sampler(
    transformer=lora_gemma,
    tokenizer=gemma_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)

# for eval_example in QUALITATIVE_EVAL_EXAMPLES:
#   question = eval_example["question"]
#   answer = eval_example["answer"]
#   response = generate(
#       question,
#       sampler,
#       temperature=INFERENCE_TEMPERATURE,
#       top_k=INFERENCE_TOP_K,
#       top_p=INFERENCE_TOP_P,
#   )

#   print(f"Question:\n{question}")
#   print(f"Answer:\n{answer}")
#   print(f"Response:\n{response}")
#   print("===============")

