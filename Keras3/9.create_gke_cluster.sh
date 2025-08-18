set -euo pipefail

# --- Configuration ---
CONFIG_FILE="${1:-config.conf}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found at '$CONFIG_FILE'" >&2
    exit 1
fi

source "$CONFIG_FILE"

# --- Execution ---
echo "Creating GKE Autopilot cluster using config from '${CONFIG_FILE}'..."
gcloud container clusters create-auto "${CLUSTER_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --release-channel=rapid
echo "Getting kubectl credentials..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" --project "${PROJECT_ID}" --region "${REGION}"
echo "Creating kubectl secret..."
kubectl create secret generic hf-secret  --from-literal=hf_api_token="${HF_TOKEN}"  --dry-run=client -o yaml | kubectl apply -f -
echo "Done."