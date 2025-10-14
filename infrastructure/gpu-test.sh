#!/bin/bash

# GPU Test Script
# Tests GPU functionality in the Kubernetes cluster

set -e

echo "=== GPU Test Script ==="

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

echo ""
echo "Step 1: Verifying GPU resources are available..."
GPU_AVAILABLE=$(kubectl describe nodes | grep "nvidia.com/gpu" | grep -v "0$" | wc -l)
if [ "$GPU_AVAILABLE" -eq 0 ]; then
    echo "❌ No GPU resources available. Run gpu-deploy.sh first."
    exit 1
fi
echo "✅ GPU resources found"