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

# Configuration - Update these if needed
SSH_KEY_PATH="~/.ssh/aws-key-pair.pem"
SSH_USER="ubuntu"

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
./remote-gpu-setup.sh
check_success "Remote GPU node setup"

echo ""
echo "Phase 2: Device plugin deployment..."
./gpu-deploy.sh
check_success "GPU device plugin deployment"

echo ""
echo "Phase 3: GPU functionality testing..."
./gpu-test.sh
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
