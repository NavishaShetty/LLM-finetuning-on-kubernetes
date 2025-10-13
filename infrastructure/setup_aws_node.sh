#!/bin/bash
set -euo pipefail

# --- Configuration ---
# Set your AWS node's public and private IP addresses here.
# These variables will be used by the setup scripts.
PUBLIC_IP="YOUR_PUBLIC_IP_HERE"    # e.g., "54.123.45.67"
PRIVATE_IP="YOUR_PRIVATE_IP_HERE"  # e.g., "172.31.0.10"
# -------------------

echo "Starting AWS Node setup with:"
echo "  Public IP: ${PUBLIC_IP}"
echo "  Private IP: ${PRIVATE_IP}"
echo ""

# Ensure the scripts are executable
chmod +x install_kubernetes.sh complete-gpu-setup.sh

# Run Kubernetes installation
echo "--- Running Kubernetes Installation ---"
# It's assumed install_kubernetes.sh will either read these IPs from
# environment variables or accept them as arguments.
# Example if using environment variables:
PUBLIC_IP="${PUBLIC_IP}" PRIVATE_IP="${PRIVATE_IP}" ./install_kubernetes.sh
# Example if using arguments:
# ./install_kubernetes.sh "${PUBLIC_IP}" "${PRIVATE_IP}"

if [ $? -eq 0 ]; then
    echo "Kubernetes installation completed successfully."
else
    echo "ERROR: Kubernetes installation failed. Exiting."
    exit 1
fi

echo ""

# Run GPU setup
echo "--- Running GPU Setup ---"
# Similarly, complete-gpu-setup.sh is assumed to handle IPs.
# Example if using environment variables:
PUBLIC_IP="${PUBLIC_IP}" PRIVATE_IP="${PRIVATE_IP}" ./complete-gpu-setup.sh
# Example if using arguments:
# ./complete-gpu-setup.sh "${PUBLIC_IP}" "${PRIVATE_IP}"

if [ $? -eq 0 ]; then
    echo "GPU setup completed successfully."
else
    echo "ERROR: GPU setup failed. Exiting."
    exit 1
fi

echo ""
echo "AWS Node setup finished successfully!"
