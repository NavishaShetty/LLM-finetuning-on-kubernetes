#!/bin/bash

# Complete GPU Setup Script
# This script runs the entire GPU setup process

set -e

echo "=== Complete GPU Setup Script ==="
echo "This script will:"
echo "1. Configure the GPU node remotely"
echo "2. Deploy the NVIDIA device plugin"
echo "3. Test GPU functionality"
echo ""

# Configuration - IP and SSH details are expected from environment variables.
SSH_KEY_PATH="${SSH_KEY_PATH:?Error: SSH_KEY_PATH not set for complete-gpu-setup.sh}"
SSH_USER="${SSH_USER:?Error: SSH_USER not set for complete-gpu-setup.sh}"

# Get the Public IP from the environment variable passed by setup_aws_node.sh
PUBLIC_IP="${PUBLIC_IP:?Error: PUBLIC_IP not set for complete-gpu-setup.sh}"

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "nvidia-device-plugin.yaml" ]; then
    echo "❌ nvidia-device-plugin.yaml not found in current directory"
    exit 1
fi

if [ ! -f "gpu-node-setup.sh" ]; then
    echo "❌ gpu-node-setup.sh not found. Please run this script in the directory with all GPU setup scripts."
    exit 1
fi

kubectl get nodes > /dev/null 2>&1
check_success "Kubernetes cluster connection"

echo ""
echo "Phase 1: Remote GPU node configuration..."
# Pass the necessary environment variables to remote-gpu-setup.sh
GPU_NODE_IP="${PUBLIC_IP}" \
SSH_KEY_PATH="${SSH_KEY_PATH}" \
SSH_USER="${SSH_USER}" \
./infrastructure/remote-gpu-setup.sh
check_success "Remote GPU node setup"

echo ""
echo "Phase 2: Device plugin deployment..."
./infrastructure/gpu-deploy.sh
check_success "GPU device plugin deployment"

echo ""
echo "Phase 3: GPU functionality testing..."
./infrastructure/gpu-test.sh
check_success "GPU functionality test"

echo ""
echo "Complete GPU setup finished successfully!"
echo "Your Kubernetes cluster is now ready for GPU workloads."
echo ""
echo "Summary:"
kubectl describe nodes | grep nvidia.com/gpu
echo ""
echo "To run GPU workloads, use:"
echo "  resources:"
echo "    limits:"
echo "      nvidia.com/gpu: 1"
echo "  runtimeClassName: nvidia"
