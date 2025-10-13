#!/bin/bash

# Remote GPU Setup Script
# This script runs gpu-node-setup.sh on the remote GPU node via SSH

set -e

# Configuration - Update these variables
# Configuration - All necessary details are expected from environment variables.
SSH_KEY_PATH="${SSH_KEY_PATH:?Error: SSH_KEY_PATH not set for remote-gpu-setup.sh}"
SSH_USER="${SSH_USER:?Error: SSH_USER not set for remote-gpu-setup.sh}"
NODE_IP="${NODE_IP:?Error: NODE_IP not set for remote-gpu-setup.sh}"

echo "=== Remote GPU Setup Script ==="

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Expand tilde in SSH key path
SSH_KEY_PATH_EXPANDED=$(eval echo "$SSH_KEY_PATH")

echo ""
echo "Step 1: Testing SSH connection to GPU node..."
ssh -i "$SSH_KEY_PATH_EXPANDED" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SSH_USER@$NODE_IP" "echo 'SSH connection successful'"
check_success "SSH connection test"

echo ""
echo "Step 2: Copying gpu-node-setup.sh to remote node..."
scp -i "$SSH_KEY_PATH_EXPANDED" gpu-node-setup.sh "$SSH_USER@$NODE_IP:/tmp/"
check_success "Script copy to remote node"

echo ""
echo "Step 3: Making script executable on remote node..."
ssh -i "$SSH_KEY_PATH_EXPANDED" "$SSH_USER@$NODE_IP" "chmod +x /tmp/gpu-node-setup.sh"
check_success "Script permission setting"

echo ""
echo "Step 4: Running GPU setup on remote node..."
ssh -i "$SSH_KEY_PATH_EXPANDED" "$SSH_USER@$NODE_IP" "/tmp/gpu-node-setup.sh"
check_success "Remote GPU setup execution"

echo ""
echo "Step 5: Cleaning up remote script..."
ssh -i "$SSH_KEY_PATH_EXPANDED" "$SSH_USER@$NODE_IP" "rm /tmp/gpu-node-setup.sh"
check_success "Remote cleanup"

echo ""
echo "✅ Remote GPU setup completed successfully!"
