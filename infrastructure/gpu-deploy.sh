#!/bin/bash

# GPU Device Plugin Deployment Script
# Run this from your MacOS terminal after running gpu-node-setup.sh on the GPU node

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "=== GPU Device Plugin Deployment Script ==="

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Function to wait for pods to be ready
wait_for_pod_ready() {
    local namespace=$1
    local selector=$2
    local timeout=${3:-300}
    
    echo "Waiting for pod to be ready (timeout: ${timeout}s)..."
    kubectl wait --for=condition=Ready pod -l "$selector" -n "$namespace" --timeout="${timeout}s"
}

echo ""
echo "Step 1: Verifying cluster connection..."
kubectl get nodes > /dev/null 2>&1
check_success "Cluster connection verification"

echo ""
echo "Step 2: Creating NVIDIA RuntimeClass..."
cat << 'RUNTIME_EOF' | kubectl apply -f -
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
RUNTIME_EOF
check_success "NVIDIA RuntimeClass creation"

echo ""
echo "Step 3: Verifying RuntimeClass was created..."
kubectl get runtimeclass nvidia > /dev/null 2>&1
check_success "RuntimeClass verification"

echo ""
echo "Step 4: Deploying NVIDIA device plugin..."
if [ ! -f "$SCRIPT_DIR/nvidia-device-plugin.yaml" ]; then
    echo "❌ nvidia-device-plugin.yaml file not found in current directory"
    echo "Please ensure the file exists and has runtimeClassName: nvidia configured"
    exit 1
fi

kubectl apply -f "$SCRIPT_DIR/nvidia-device-plugin.yaml"
check_success "NVIDIA device plugin deployment"

echo ""
echo "Step 5: Waiting for device plugin to be ready..."
sleep 10
wait_for_pod_ready "kube-system" "name=nvidia-device-plugin-ds" 120
check_success "Device plugin pod ready"

echo ""
echo "Step 6: Verifying device plugin is running..."
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds
check_success "Device plugin status check"

echo ""
echo "Step 7: Checking device plugin logs..."
echo "Device plugin logs (last 10 lines):"
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds --tail=10

echo ""
echo "Step 8: Verifying GPU resources are available..."
sleep 20
GPU_RESOURCES=$(kubectl describe nodes | grep "nvidia.com/gpu" | head -1)
if [[ $GPU_RESOURCES == *"nvidia.com/gpu"* ]]; then
    echo "✅ GPU resources detected:"
    kubectl describe nodes | grep nvidia.com/gpu
else
    echo "❌ No GPU resources found on nodes"
    echo "Check device plugin logs for errors:"
    kubectl logs -n kube-system -l name=nvidia-device-plugin-ds
    exit 1
fi

echo ""
echo "✅ GPU device plugin deployment completed successfully!"