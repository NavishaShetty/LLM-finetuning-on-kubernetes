#!/bin/bash

# =============================================================================
# Kubernetes Setup with Kubespray on AWS G4DN Instance
# This script will install Kubespray on the local macOS machine,
# configure it, and deploy Kubernetes to a single AWS node.
# =============================================================================

# Configuration - IP addresses are expected from environment variables set by setup_aws_node.sh
AWS_INSTANCE_IP="${PUBLIC_IP:?Error: PUBLIC_IP not set for install_kubernetes.sh}"
AWS_INSTANCE_PRIVATE_IP="${PRIVATE_IP:?Error: PRIVATE_IP not set for install_kubernetes.sh}"
SSH_KEY_PATH="${SSH_KEY_PATH:?Error: SSH_KEY_PATH not set for install_kubernetes.sh}"
SSH_USER="${SSH_USER:?Error: SSH_USER not set for install_kubernetes.sh}"

# Global path definitions
SCRIPT_REAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
KUBESPRAY_ROOT_DIR="$SCRIPT_REAL_DIR/kubespray"

# =============================================================================
# SECTION 1: TEST AWS MACHINE CONNECTION
# =============================================================================
echo "=== SECTION 1: Testing AWS Machine Connection ==="

# Test SSH connection to AWS instance
test_aws_connection() {
    echo "Testing SSH connection to AWS instance..."
    ssh -i $SSH_KEY_PATH -o ConnectTimeout=10 -o StrictHostKeyChecking=no $SSH_USER@$AWS_INSTANCE_IP "echo 'Connection successful! Instance details:' && uname -a && free -h && df -h"
    
    if [ $? -eq 0 ]; then
        echo "✅ AWS instance is accessible"
    else
        echo "❌ Cannot connect to AWS instance. Check:"
        exit 1
    fi
}

# Execute this function
test_aws_connection

# =============================================================================
# SECTION 2: INSTALL KUBESPRAY ON LOCAL MACOS MACHINE
# =============================================================================
echo "=== SECTION 2: Installing Kubespray on macOS ==="

install_kubespray_macos() {
    echo "Installing prerequisites on macOS..."
    
    # Install Homebrew if not present
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install Python 3 and pip
    echo "Installing Python 3..."
    brew install python3
    
    # Install git if not present
    brew install git
    
    # Ensure we are in the script's real directory for cloning
    ( # Start a subshell to localize cd commands
    cd "$SCRIPT_REAL_DIR"
    
    # Check if kubespray directory exists and remove it
    if [ -d "kubespray" ]; then
        echo "Kubespray directory already exists at $KUBESPRAY_ROOT_DIR, removing..."
        rm -rf kubespray
        echo "✅ Removed existing kubespray directory"
    fi
    
    # Clone Kubespray repository
    echo "Cloning Kubespray repository to $KUBESPRAY_ROOT_DIR..."
    git clone https://github.com/kubernetes-sigs/kubespray.git
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully cloned Kubespray"
    else
        echo "❌ Failed to clone Kubespray"
        exit 1
    fi

    # Change into the kubespray directory to setup venv
    cd "$KUBESPRAY_ROOT_DIR"
    
    # Setup Python virtual environment for Kubespray
    echo "Setting up Python virtual environment in $KUBESPRAY_ROOT_DIR..."
    
    # Create virtual environment
    python3 -m venv kubespray-venv
    
    # Activate virtual environment (temporarily for installation in subshell)
    source kubespray-venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Ansible
    echo "Installing Ansible in virtual environment..."
    pip install ansible
    
    # Install Kubespray requirements
    echo "Installing Kubespray Python requirements..."
    pip install -r requirements.txt
    
    echo "✅ Kubespray installation completed in subshell."
    echo "Kubespray installed in: $KUBESPRAY_ROOT_DIR"
    
    # Verify installation (within subshell context)
    echo ""
    echo "=== VERIFYING KUBESPRAY INSTALLATION ==="
    ansible --version
    python3 -c "import ansible; print('Ansible Python module: OK')"
    
    if [ $? -eq 0 ]; then
        echo "✅ Kubespray installation verification successful!"
    else
        echo "❌ Kubespray installation verification failed in subshell"
        echo "Please check the error messages above"
        exit 1 # Exit subshell on failure
    fi
    ) # End of subshell

    # After subshell, activate venv in main shell for subsequent sections
    echo "Activating Kubespray virtual environment in main shell: $KUBESPRAY_ROOT_DIR/kubespray-venv/bin/activate"
    source "$KUBESPRAY_ROOT_DIR/kubespray-venv/bin/activate"

    # Now change to the Kubespray root directory for the remainder of the script
    echo "Changing current directory to Kubespray root: $KUBESPRAY_ROOT_DIR"
    cd "$KUBESPRAY_ROOT_DIR"

    echo "You can now proceed to section 3 (configure_kubespray)"
}

# Execute this function
install_kubespray_macos

# =============================================================================
# SECTION 3: CONFIGURE KUBESPRAY FOR AWS INSTANCE
# =============================================================================
echo "=== SECTION 3: Configuring Kubespray for AWS Instance ==="

configure_kubespray() {
    echo "Configuring Kubespray for single-node AWS deployment..."
    
    # We assume the script is already in KUBESPRAY_ROOT_DIR and venv is activated.
    # Check if inventory directory exists (part of kubespray clone)
    if [ ! -d "inventory/sample" ]; then
        echo "❌ Kubespray inventory/sample directory not found. Expected in current directory: $(pwd)"
        echo "Ensure Kubespray was cloned correctly and the script is running from \$KUBESPRAY_ROOT_DIR."
        exit 1
    fi
    
    # Copy sample inventory
    echo "Creating inventory configuration..."
    cp -rfp inventory/sample inventory/mycluster
    
    # Create inventory file for single node
    cat > inventory/mycluster/inventory.ini << EOF
[all]
k8s-node1 ansible_host=$AWS_INSTANCE_IP ip=$AWS_INSTANCE_PRIVATE_IP ansible_user=$SSH_USER

[kube_control_plane]
k8s-node1

[etcd]
k8s-node1

[kube_node]
k8s-node1

[calico_rr]

[k8s_cluster:children]
kube_control_plane
kube_node
calico_rr
EOF

    # Configure SSH settings
    cat > inventory/mycluster/group_vars/all/ansible.yml << EOF
---
# Ansible settings
ansible_ssh_private_key_file: $SSH_KEY_PATH
ansible_ssh_common_args: '-o StrictHostKeyChecking=no'
ansible_python_interpreter: /usr/bin/python3
EOF

    # Copy the default k8s-cluster.yml from sample to avoid version issues
    echo "Using default Kubernetes configuration from sample..."
    cp inventory/sample/group_vars/k8s_cluster/k8s-cluster.yml inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml
    
    # Make minimal modifications for single node
    cat >> inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml << EOF

# Single node configuration additions
# Allow scheduling pods on control plane (single node setup)
kubeadm_control_plane_endpoint: "$AWS_INSTANCE_PRIVATE_IP:6443"

# Enable metrics server
metrics_server_enabled: true

# Container runtime
container_manager: containerd
EOF

    # Configure add-ons
    cat > inventory/mycluster/group_vars/k8s_cluster/addons.yml << EOF
---
# Kubernetes addons
dashboard_enabled: false
helm_enabled: true
registry_enabled: false
local_volume_provisioner_enabled: true
cephfs_provisioner_enabled: false
rbd_provisioner_enabled: false
ingress_nginx_enabled: true
cert_manager_enabled: true
metallb_enabled: false
metrics_server_enabled: true

# Storage class
local_volume_provisioner_storage_classes:
  fast-disks:
    host_dir: /mnt/fast-disks
    mount_dir: /mnt/fast-disks
  slow-disks:
    host_dir: /mnt/slow-disks
    mount_dir: /mnt/slow-disks
EOF

    echo "✅ Kubespray configuration completed"
    echo "Configuration files created in: $KUBESPRAY_ROOT_DIR/inventory/mycluster/"
    echo ""
    echo "Configuration summary:"
    echo "- Kubespray directory: $KUBESPRAY_ROOT_DIR"
}

# Execute this function
configure_kubespray

# =============================================================================
# SECTION 4: DEPLOY KUBERNETES USING KUBESPRAY
# =============================================================================
echo "=== SECTION 4: Deploying Kubernetes ==="

deploy_kubernetes() {
    echo "Starting Kubernetes deployment with Kubespray..."
    
    # We assume the script is already in KUBESPRAY_ROOT_DIR and venv is activated.
    
    # Verify inventory exists
    if [ ! -f "inventory/mycluster/inventory.ini" ]; then
        echo "❌ Inventory file not found. Please run configure_kubespray first"
        exit 1
    fi
    
    # Test connection first
    echo "Testing connection to remote machine..."
    ansible -i inventory/mycluster/inventory.ini all -m ping --become
    
    if [ $? -ne 0 ]; then
        echo "❌ Cannot connect to remote machine."
        return 1
    fi
    
    echo "✅ Connection test successful!"

    # Deploy with retry on apt lock failures
    echo "Deploying Kubernetes cluster (this may take 15-30 minutes)..."
    echo "This will handle OS preparation and Kubernetes installation in one step..."
    
    DEPLOY_SUCCESS=false
    RETRY_COUNT=0
    MAX_RETRIES=3
    
    while [ $DEPLOY_SUCCESS = false ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "Deployment attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES..."
        
        ansible-playbook -i inventory/mycluster/inventory.ini \
            --become --become-user=root \
            cluster.yml
        
        if [ $? -eq 0 ]; then
            DEPLOY_SUCCESS=true
            echo "✅ Kubernetes deployment completed successfully!"
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo "❌ Deployment failed, waiting 2 minutes before retry..."
                echo "Clearing apt locks on remote machine..."
                ansible -i inventory/mycluster/inventory.ini all -m shell \
                    -a "sudo killall -9 unattended-upgr 2>/dev/null || true; sudo systemctl stop unattended-upgrades" \
                    --become
                sleep 120
            else
                echo "❌ Kubernetes deployment failed after $MAX_RETRIES attempts"
                return 1
            fi
        fi
    done
    
    # If we reached here, deployment was successful (or max retries reached and returned 1)
    if [ "$DEPLOY_SUCCESS" = true ]; then
        echo "✅ Deployment complete!"
        return 0
    else
        # Should already have returned 1, but explicitly for clarity
        return 1
    fi
}

# Execute this function
deploy_kubernetes

# =============================================================================
# SECTION 5: INSTALL AND CONFIGURE KUBECTL
# =============================================================================
echo "=== SECTION 5: Installing and Configuring kubectl ==-"

echo "Using SSH key: $SSH_KEY_PATH"
echo "Connecting to: $SSH_USER@$AWS_INSTANCE_IP"

# Test SSH connection first
echo "=== TESTING SSH CONNECTION ==-"
ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SSH_USER@$AWS_INSTANCE_IP" "echo 'SSH connection successful!'"

if [ $? -ne 0 ]; then
    echo "❌ SSH connection failed"
    exit 1
fi

echo "✅ SSH connection successful!"

# Check if admin.conf exists and get its location
echo ""
echo "=== FINDING KUBECONFIG ON REMOTE MACHINE ==-"
echo "Checking for kubeconfig files on remote machine..."

ssh -i "$SSH_KEY_PATH" "$SSH_USER@$AWS_INSTANCE_IP" "
echo 'Looking for kubeconfig files...'
sudo find /etc/kubernetes -name '*.conf' 2>/dev/null || echo 'No kubeconfig found in /etc/kubernetes'
echo ''
echo 'Checking if cluster is running...'
sudo systemctl status kubelet | head -5
echo ''
echo 'Checking for running containers...'
sudo docker ps 2>/dev/null | head -5 || sudo crictl ps 2>/dev/null | head -5 || echo 'No containers found'
"

# Copy kubeconfig from remote machine
echo ""
echo "=== COPYING KUBECONFIG FROM REMOTE MACHINE ==-"
echo "Copying kubeconfig with proper sudo access..."

# Ensure .kube directory exists in user's home
mkdir -p "$HOME/.kube"

# Backup existing config if it exists
if [ -f "$HOME/.kube/config" ]; then
    echo "Backing up existing kubeconfig..."
    cp "$HOME/.kube/config" "$HOME/.kube/config.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Copy the kubeconfig with proper sudo permissions
echo "Copying admin.conf from remote machine..."
ssh -i "$SSH_KEY_PATH" "$SSH_USER@$AWS_INSTANCE_IP" "sudo cat /etc/kubernetes/admin.conf" > "$HOME/.kube/config"

if [ $? -eq 0 ] && [ -s "$HOME/.kube/config" ]; then
    echo "✅ Successfully copied kubeconfig"
else
    echo "❌ Failed to copy kubeconfig or file is empty"
    exit 1
fi

# Update kubeconfig for external access
echo ""
echo "=== CONFIGURING KUBECTL FOR EXTERNAL ACCESS ==-"
echo "Updating kubeconfig to use external IP address..."

# Replace internal IP addresses with external IP
sed -i.bak "s/127.0.0.1:6443/$AWS_INSTANCE_IP:6443/g" "$HOME/.kube/config"
sed -i.bak2 "s/${AWS_INSTANCE_PRIVATE_IP}:6443/${AWS_INSTANCE_IP}:6443/g" "$HOME/.kube/config"

echo "Updated server endpoint to use external IP: $AWS_INSTANCE_IP:6443"

# Configure TLS settings for external access
echo "Setting insecure-skip-tls-verify for external cluster access..."
kubectl config set-cluster cluster.local --server=https://$AWS_INSTANCE_IP:6443 --insecure-skip-tls-verify=true --kubeconfig="$HOME/.kube/config"

echo "✅ Configured kubectl for external cluster access"

# Test kubectl connection
echo ""
echo "=== TESTING KUBECTL CONNECTION ==-"
echo "Testing cluster connection..."
kubectl get nodes --kubeconfig="$HOME/.kube/config"

if [ $? -eq 0 ]; then
    echo "✅ kubectl successfully connected to cluster!"
    echo ""
    echo "Cluster details:"
    kubectl get nodes -o wide --kubeconfig="$HOME/.kube/config"
    echo ""
    echo "Cluster info:"
    kubectl cluster-info --kubeconfig="$HOME/.kube/config"
    return 0 # Indicate success
else
    echo "❌ kubectl connection failed"
    echo ""
    echo "HINT: Please ensure port 6443 is open in your AWS security group."
    echo "Recommended rule:"
    echo "• Type: Custom TCP, Port: 6443, Source: $(curl -s -4 ifconfig.me)/32"
    echo ""
    echo "You can test the connection manually from your local machine with:"
    echo "kubectl get nodes --insecure-skip-tls-verify=true --kubeconfig=\"$HOME/.kube/config\""
    return 1 # Indicate failure
fi
