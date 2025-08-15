import os
import subprocess
import sys
import datetime
from google.cloud import aiplatform
from typing import Tuple
from utils import load_config

# Load configuration from config.conf into environment variables
load_config()

# --- Configuration from Environment ---
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET_URI = os.environ.get("BUCKET_URI")
MODEL_NAME = os.environ.get("MODEL_NAME")
HUGGINGFACE_MODEL_DIR = os.environ.get("HUGGINGFACE_MODEL_DIR")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")
FT_HF_MODEL = os.environ.get("FT_HF_MODEL")

if not all([PROJECT_ID, REGION, BUCKET_URI, MODEL_NAME, HUGGINGFACE_MODEL_DIR, FT_HF_MODEL, HF_MODEL_ID]):
    print("❌ Error: One or more required variables (PROJECT_ID, REGION, BUCKET_URI, MODEL_NAME, HUGGINGFACE_MODEL_DIR, FT_HF_MODEL, HF_MODEL_ID) not found in config.conf.")
    sys.exit(1)

# Converted model
# Deployed model
DEPLOYED_MODEL_URI = f"{BUCKET_URI}/{MODEL_NAME}"
MODEL_NAME_HEXLLM = f"{MODEL_NAME}-hexllm"
MODEL_ID= f"{BUCKET_URI}/{MODEL_NAME}"
TPU_DEPLOYMENT_REGION=REGION

# Create the service account for the Vertex AI endpoint
SERVICE_ACCOUNT_NAME = "gemma-vertexai"
SERVICE_ACCOUNT_DISPLAY_NAME = "Gemma Vertex AI endpoint"
SERVICE_ACCOUNT = f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com"
# Or use an existing one
# SERVICE_ACCOUNT = "" # @param {type:"string"}
assert SERVICE_ACCOUNT.endswith(f"@{PROJECT_ID}.iam.gserviceaccount.com"), \
    "SERVICE_ACCOUNT must end with @<PROJECT_ID>.iam.gserviceaccount.com"

def run_gcloud_command(command_string, error_message):
    """
    Helper function to run gcloud commands as a string with shell=True
    and handle errors. This is suitable for commands needing shell interpretation (e.g., format strings).
    """
    try:
        # Using check_call with shell=True allows for complex command strings
        # and relies on the shell for argument parsing (e.g., quotes).
        subprocess.check_call(command_string, shell=True)
        print(f"✔️ Command executed successfully: {command_string}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {error_message}")
        print(f"Command: {command_string}")
        print(f"Error details: {e}")
        sys.exit(1) # Exit the script on error

# Check if the service account exists
print(f"Checking for service account: {SERVICE_ACCOUNT}")
try:
    # Use subprocess.run with shell=True and careful quoting for the format string.
    # The --format "value(email)" requires shell interpretation.
    result = subprocess.run(
        f'gcloud iam service-accounts describe {SERVICE_ACCOUNT} --format "value(email)"',
        shell=True,
        capture_output=True,
        text=True # Decode stdout/stderr as text
    )

    if result.returncode == 0:
        print("✔️ The service account exists")
    else:
        # If the describe command failed, it likely means the service account does not exist.
        print("⚙️ Creating the service account…")

        # Crucially, ensure the display name is correctly quoted within the command string.
        # This handles spaces in the display name when shell=True is used.
        create_command = (
            f'gcloud iam service-accounts create {SERVICE_ACCOUNT_NAME} '
            f'--display-name "{SERVICE_ACCOUNT_DISPLAY_NAME}"' # Correctly quoted
        )
        run_gcloud_command(create_command, "Failed to create service account.")

        # Grant "Storage Object Admin" role
        storage_role_command = (
            f'gcloud projects add-iam-policy-binding {PROJECT_ID} '
            f'--member "serviceAccount:{SERVICE_ACCOUNT}" '
            f'--role "roles/storage.objectAdmin"'
        )
        run_gcloud_command(storage_role_command, "Failed to grant Storage Object Admin role.")

        # Grant "Vertex AI User" role
        vertex_ai_role_command = (
            f'gcloud projects add-iam-policy-binding {PROJECT_ID} '
            f'--member "serviceAccount:{SERVICE_ACCOUNT}" '
            f'--role "roles/aiplatform.user"'
        )
        run_gcloud_command(vertex_ai_role_command, "Failed to grant Vertex AI User role.")
        print("✔️ Service account created and roles granted.")

except FileNotFoundError:
    print("❌ Error: 'gcloud' command not found. Please ensure Google Cloud SDK is installed and configured.")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    sys.exit(1)


#
# Initialize Vertex AI
#

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

# Upload model to GCS

run_gcloud_command(f"gcloud storage rsync --recursive --verbosity error {HUGGINGFACE_MODEL_DIR} {DEPLOYED_MODEL_URI}", "Failed to upload model")

# Define helper functions to deploy to vLLM container

HEXLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai-restricted/vertex-vision-model-garden-dockers/hex-llm-serve:20241210_2323_RC00"

# Sets ct5lp-hightpu-4t (4 TPU chips) to deploy models.
machine_type = "ct5lp-hightpu-4t"  # @param ["ct5lp-hightpu-4t", "ct5lp-hightpu-8t"]
# Note: 1 TPU V5 chip has only one core.
tpu_type = "TPU_V5e"

# @markdown Set enable_prefix_cache_hbm to False if you don't want to use [prefix caching](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-hex-llm#prefix-caching).
enable_prefix_cache_hbm = True  # @param {type:"boolean"}

# @markdown Set the disaggregated topology to balance the TTFT and TPOT.
# @markdown This is an **experimental** feature and is only supported for single host deployments.
# @markdown If want to enable the feature, set this parameter to a string of the form `"num_prefill_workers,num_decode_workers"`, like `"3,1"`.
disagg_topo = None  # @param

if "7" in MODEL_ID:
    tpu_count = 4
    tpu_topo = "1x4"
elif "2" in MODEL_ID:
    tpu_count = 1
    tpu_topo = "1x1"
else:
    raise ValueError(f"Unsupported MODEL_ID: {MODEL_ID}")


# Server parameters.
tensor_parallel_size = tpu_count

# @markdown Set the server parameters.

# Fraction of HBM memory allocated for KV cache after model loading. A larger value improves throughput but gives higher risk of TPU out-of-memory errors with long prompts.
hbm_utilization_factor = 0.8  # @param
# Maximum number of running sequences in a continuous batch.
max_running_seqs = 256  # @param
# Maximum context length for a request.
max_model_len = 4096  # @param

# Endpoint configurations.
min_replica_count = 1
max_replica_count = 1


def get_job_name_with_datetime(prefix: str) -> str:
        suffix = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        return f"{prefix}{suffix}"


def deploy_model_hexllm(
    model_name: str,
    model_id: str,
    publisher: str,
    publisher_model_id: str,
    base_model_id: str = None,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 4,
    machine_type: str = "ct5lp-hightpu-4t",
    tpu_topology: str = "1x4",
    disagg_topology: str = None,
    hbm_utilization_factor: float = 0.8,
    max_running_seqs: int = 256,
    max_model_len: int = 4096,
    enable_prefix_cache_hbm: bool = False,
    endpoint_id: str = "",
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    use_dedicated_endpoint: bool = False,
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    """Deploys models with Hex-LLM on TPU in Vertex AI."""
    if endpoint_id:
        aip_endpoint_name = (
            f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{endpoint_id}"
        )
        endpoint = aiplatform.Endpoint(aip_endpoint_name)
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=f"{model_name}-endpoint",
            location=TPU_DEPLOYMENT_REGION,
            dedicated_endpoint_enabled=use_dedicated_endpoint,
        )

    if not base_model_id:
        base_model_id = model_id

    if not tensor_parallel_size:
        tensor_parallel_size = int(machine_type[-2])

    num_hosts = int(tpu_topology.split("x")[0])

    # Learn more about the supported arguments and environment variables at https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-hex-llm#config-server.
    hexllm_args = [
        "--host=0.0.0.0",
        "--port=7080",
        f"--model={model_id}",
        f"--data_parallel_size={data_parallel_size}",
        f"--tensor_parallel_size={tensor_parallel_size}",
        f"--num_hosts={num_hosts}",
        f"--hbm_utilization_factor={hbm_utilization_factor}",
        f"--max_running_seqs={max_running_seqs}",
        f"--max_model_len={max_model_len}",
    ]
    if disagg_topology:
        hexllm_args.append(f"--disagg_topo={disagg_topology}")
    if enable_prefix_cache_hbm and not disagg_topology:
        hexllm_args.append("--enable_prefix_cache_hbm")

    env_vars = {
        "MODEL_ID": base_model_id,
        "HEX_LLM_LOG_LEVEL": "info",
        "DEPLOY_SOURCE": "notebook",
    }

    # HF_TOKEN is not a compulsory field and may not be defined.
    try:
        if HF_TOKEN:
            env_vars.update({"HF_TOKEN": HF_TOKEN})
    except:
        pass

    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=HEXLLM_DOCKER_URI,
        serving_container_command=["python", "-m", "hex_llm.server.api_server"],
        serving_container_args=hexllm_args,
        serving_container_ports=[7080],
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_environment_variables=env_vars,
        serving_container_shared_memory_size_mb=(16 * 1024),  # 16 GB
        serving_container_deployment_timeout=7200,
        location=TPU_DEPLOYMENT_REGION,
    )

    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        tpu_topology=tpu_topology if num_hosts > 1 else None,
        deploy_request_timeout=1800,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
    )
    return model, endpoint



# Deploy model

LABEL = "hexllm_tpu"
model, endpoint = deploy_model_hexllm(
    model_name=get_job_name_with_datetime(prefix=MODEL_NAME_HEXLLM),
    model_id=FT_HF_MODEL, #MODEL_ID,
    publisher="google",
    publisher_model_id="gemma3-1b-instruct",
    base_model_id=HF_MODEL_ID,
    tensor_parallel_size=tensor_parallel_size,
    machine_type=machine_type,
    tpu_topology=tpu_topo,
    disagg_topology=disagg_topo,
    hbm_utilization_factor=hbm_utilization_factor,
    max_running_seqs=max_running_seqs,
    max_model_len=max_model_len,
    enable_prefix_cache_hbm=enable_prefix_cache_hbm,
    min_replica_count=min_replica_count,
    max_replica_count=max_replica_count,
    use_dedicated_endpoint=False,
)

# Online inference

TEST_EXAMPLES = [
        #"Lizzy has to ship 540 pounds of fish that are packed into 30-pound crates. If the shipping cost of each crate is $1.5, how much will Lizzy pay for the shipment?",
        #"A school choir needs robes for each of its 30 singers. Currently, the school has only 12 robes so they decided to buy the rest. If each robe costs $2, how much will the school spend?",
        "Peter has 25 apples to sell. He sells the first 10 for $1 each, 10 more for $0.75 each and the last 5 for $0.50 each. How much money does he make?",
        "Bea has $40. She wants to rent a bike for $4/hour. How many hours can she ride the bike?",
        ]



# Prompt template for the training data and the finetuning tests
PROMPT_TEMPLATE = "user: {instruction}\nmodel: {response}\n"

TEST_PROMPTS = [
        PROMPT_TEMPLATE.format(instruction=example, response="")
        for example in TEST_EXAMPLES
        ]

def test_vertexai_endpoint(endpoint: aiplatform.Endpoint):
    for question, prompt in zip(TEST_EXAMPLES, TEST_PROMPTS):
        instance = {
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 10,
                "raw_response": True,
                }
        response = endpoint.predict(instances=[instance])
        output = response.predictions[0]
        print(f"{question}\n{output}\n{'- '*40}")


test_vertexai_endpoint(endpoint)
