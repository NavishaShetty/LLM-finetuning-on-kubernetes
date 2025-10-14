#!/bin/bash
set -euo pipefail

# --- Configuration ---
# Set your AWS node's public and private IP addresses here.
# These variables will be used by the setup scripts.
PUBLIC_IP="13.58.119.178"       # e.g., "54.123.45.67"
PRIVATE_IP="172.31.42.162"    # e.g., "172.31.0.10"
SSH_KEY_PATH="~/.ssh/aws-key-pair.pem" # Path to your AWS SSH key
SSH_USER="ubuntu"           # SSH user for your AWS instance
# -------------------

echo "Starting AWS Node setup with:"
echo "  Public IP: ${PUBLIC_IP}"
echo "  Private IP: ${PRIVATE_IP}"
echo ""

# Ensure the scripts are executable
chmod +x infrastructure/install_kubernetes.sh infrastructure/complete-gpu-setup.sh

# Run Kubernetes installation
echo "--- Running Kubernetes Installation ---"
PUBLIC_IP="${PUBLIC_IP}" \
PRIVATE_IP="${PRIVATE_IP}" \
SSH_KEY_PATH="${SSH_KEY_PATH}" \
SSH_USER="${SSH_USER}" \
bash ./infrastructure/install_kubernetes.sh

if [ $? -eq 0 ]; then
    echo "Kubernetes installation completed successfully."
else
    echo "ERROR: Kubernetes installation failed. Exiting."
    exit 1
fi

echo ""

# Run GPU setup
echo "--- Running GPU Setup ---"
PUBLIC_IP="${PUBLIC_IP}" \
PRIVATE_IP="${PRIVATE_IP}" \
SSH_KEY_PATH="${SSH_KEY_PATH}" \
SSH_USER="${SSH_USER}" \
bash ./infrastructure/complete-gpu-setup.sh

if [ $? -eq 0 ]; then
    echo "GPU setup completed successfully."
else
    echo "ERROR: GPU setup failed. Exiting."
    exit 1
fi

echo ""
echo "AWS Node setup finished successfully!"
