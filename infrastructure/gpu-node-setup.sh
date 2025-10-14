#!/bin/bash

# GPU Node Setup Script
# This script configures the GPU node for Kubernetes GPU support

set -e

echo "=== GPU Node Setup Script ==="
echo "Configuring NVIDIA Container Toolkit and containerd..."

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo ""
echo "Step 1: Verifying NVIDIA drivers..."
nvidia-smi > /dev/null 2>&1
check_success "NVIDIA driver verification"

echo ""
echo "Step 2: Checking NVIDIA Container Toolkit..."
nvidia-ctk --version > /dev/null 2>&1
check_success "NVIDIA Container Toolkit check"

echo ""
echo "Step 3: Configuring containerd for NVIDIA runtime..."
$SUDO nvidia-ctk runtime configure --runtime=containerd
check_success "containerd NVIDIA runtime configuration"

echo ""
echo "Step 4: Generating NVIDIA CDI configuration..."
$SUDO nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
check_success "NVIDIA CDI configuration generation"

echo ""
echo "Step 5: Restarting containerd..."
$SUDO systemctl restart containerd
check_success "containerd restart"

echo ""
echo "Step 6: Restarting kubelet..."
$SUDO systemctl restart kubelet
check_success "kubelet restart"

echo ""
echo "Step 7: Verifying services are running..."
$SUDO systemctl is-active containerd > /dev/null 2>&1
check_success "containerd service verification"

$SUDO systemctl is-active kubelet > /dev/null 2>&1
check_success "kubelet service verification"

echo ""
echo "✅ GPU node setup completed successfully!"