#!/bin/bash

# =============================================================================
# Kubernetes Setup with Kubespray on AWS G4DN Instance
# Execute each section one by one as indicated
# =============================================================================

# Configuration - IP addresses are expected from environment variables set by setup_aws_node.sh
AWS_INSTANCE_IP="${PUBLIC_IP:?Error: PUBLIC_IP not set for install_kubernetes.sh}"
AWS_INSTANCE_PRIVATE_IP="${PRIVATE_IP:?Error: PRIVATE_IP not set for install_kubernetes.sh}"
SSH_KEY_PATH="${SSH_KEY_PATH:?Error: SSH_KEY_PATH not set for install_kubernetes.sh}"
SSH_USER="${SSH_USER:?Error: SSH_USER not set for install_kubernetes.sh}"

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
    
    # Get the directory where this script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    echo "Script directory: $SCRIPT_DIR"
    
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
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Check if kubespray directory exists and remove it
    if [ -d "kubespray" ]; then
        echo "Kubespray directory already exists in script directory, removing..."
        rm -rf kubespray
        echo "✅ Removed existing kubespray directory"
    fi
    
    # Clone Kubespray repository in current directory
    echo "Cloning Kubespray repository to script directory..."
    git clone https://github.com/kubernetes-sigs/kubespray.git
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully cloned Kubespray"
        cd kubespray
    else
        echo "❌ Failed to clone Kubespray"
        exit 1
    fi
    
    # Setup Python virtual environment for Kubespray
    echo "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv kubespray-venv
    
    # Activate virtual environment
    source kubespray-venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Ansible
    echo "Installing Ansible in virtual environment..."
    pip install ansible
    
    # Install Kubespray requirements
    echo "Installing Kubespray Python requirements..."
    pip install -r requirements.txt
    
    echo "✅ Kubespray installation completed"
    echo "Current directory: $(pwd)"
    echo "Kubespray installed in: $SCRIPT_DIR/kubespray"
    echo ""
    echo "IMPORTANT: Before proceeding to section 3, run:"
    echo "cd $SCRIPT_DIR/kubespray && source kubespray-venv/bin/activate"
    
    # Verify installation
    echo ""
    echo "=== VERIFYING INSTALLATION ==="
    ansible --version
    python3 -c "import ansible; print('Ansible Python module: OK')"
    
    if [ $? -eq 0 ]; then
        echo "✅ Installation verification successful!"
        echo "You can now proceed to section 3 (configure_kubespray)"
    else
        echo "❌ Installation verification failed"
        echo "Please check the error messages above"
    fi
}

# Execute this function
install_kubespray_macos

# =============================================================================
# SECTION 3: CONFIGURE KUBESPRAY FOR AWS INSTANCE
# =============================================================================
echo "=== SECTION 3: Configuring Kubespray for AWS Instance ==="

configure_kubespray() {
    echo "Configuring Kubespray for single-node AWS deployment..."
    
    # Get the directory where this script is located
    KUBESPRAY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    
    # Check if kubespray directory exists
    if [ ! -d "$KUBESPRAY_DIR" ]; then
        echo "❌ Kubespray directory not found at: $KUBESPRAY_DIR"
        echo "Please run 02_install_kubespray.sh first"
        exit 1
    fi
    
    # Activate virtual environment if not already active
    if [[ "$VIRTUAL_ENV" != *"kubespray-venv"* ]]; then
        echo "Activating virtual environment..."
        source kubespray-venv/bin/activate
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
    echo "Configuration files created in: $KUBESPRAY_DIR/inventory/mycluster/"
    echo ""
    echo "Configuration summary:"
    echo "- Kubespray directory: $KUBESPRAY_DIR"
}

# Execute this function
configure_kubespray

# =============================================================================
# SECTION 4: DEPLOY KUBERNETES USING KUBESPRAY
# =============================================================================
echo "=== SECTION 4: Deploying Kubernetes ==="

deploy_kubernetes() {
    echo "Starting Kubernetes deployment with Kubespray..."
    
    # Get the directory where this script is located
    KUBESPRAY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    
    # Check if kubespray directory exists
    if [ ! -d "$KUBESPRAY_DIR" ]; then
        echo "❌ Kubespray directory not found at: $KUBESPRAY_DIR"
        exit 1
    fi
    
    # Activate virtual environment if not already active
    if [[ "$VIRTUAL_ENV" != *"kubespray-venv"* ]]; then
        echo "Activating virtual environment..."
        source kubespray-venv/bin/activate
    fi
    
    # Verify inventory exists
    if [ ! -f "inventory/mycluster/inventory.ini" ]; then
        echo "❌ Inventory file not found. Please run 03_configure_kubespray.sh first"
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
    
    # Deploy Kubernetes cluster (modern Kubespray approach - no separate bootstrap needed)
    echo "Deploying Kubernetes cluster (this may take 15-30 minutes)..."
    echo "This will handle OS preparation and Kubernetes installation in one step..."
    
    ansible-playbook -i inventory/mycluster/inventory.ini \
        --become --become-user=root \
        cluster.yml
    
    if [ $? -eq 0 ]; then
        echo "✅ Kubernetes deployment completed successfully!"
        
        # Copy kubeconfig from remote machine
        echo "Copying kubeconfig from remote machine..."
        mkdir -p ~/.kube
        scp -i $SSH_KEY_PATH $SSH_USER@$AWS_INSTANCE_IP:/etc/kubernetes/admin.conf ~/.kube/config
        
        # Update kubeconfig to use public IP for external access
        # The cluster will use private IP internally, but we need public IP for kubectl access
        echo "Configuring kubeconfig for external access..."
        sed -i.bak "s/127.0.0.1:6443/$AWS_INSTANCE_IP:6443/g" ~/.kube/config
        sed -i.bak2 "s/172.31.13.107:6443/$AWS_INSTANCE_IP:6443/g" ~/.kube/config
        
        echo "Kubeconfig copied and configured for external access"
        echo "✅ Cluster accessible via public IP: $AWS_INSTANCE_IP:6443"
        echo ""
        echo "✅ Deployment complete!"
    else
        echo "❌ Kubernetes deployment failed"
        echo "Check the Ansible output above for specific errors"
        return 1
    fi
}

# Execute this function
deploy_kubernetes

# =============================================================================
# SECTION 5: INSTALL AND CONFIGURE KUBECTL
# =============================================================================
echo "=== SECTION 5: Installing and Configuring kubectl ==="

echo "Using SSH key: $SSH_KEY_PATH"
echo "Connecting to: $SSH_USER@$AWS_INSTANCE_IP"

# Test SSH connection first
echo "=== TESTING SSH CONNECTION ==="
ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SSH_USER@$AWS_INSTANCE_IP" "echo 'SSH connection successful!'"

if [ $? -ne 0 ]; then
    echo "❌ SSH connection failed"
    exit 1
fi

echo "✅ SSH connection successful!"

# Check if admin.conf exists and get its location
echo ""
echo "=== FINDING KUBECONFIG ON REMOTE MACHINE ==="
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
echo "=== COPYING KUBECONFIG FROM REMOTE MACHINE ==="
echo "Copying kubeconfig with proper sudo access..."

# Ensure .kube directory exists
mkdir -p ~/.kube

# Backup existing config if it exists
if [ -f ~/.kube/config ]; then
    echo "Backing up existing kubeconfig..."
    cp ~/.kube/config ~/.kube/config.backup.$(date +%Y%m%d_%H%M%S)
fi

# Copy the kubeconfig with proper sudo permissions
echo "Copying admin.conf from remote machine..."
ssh -i "$SSH_KEY_PATH" "$SSH_USER@$AWS_INSTANCE_IP" "sudo cat /etc/kubernetes/admin.conf" > ~/.kube/config

if [ $? -eq 0 ] && [ -s ~/.kube/config ]; then
    echo "✅ Successfully copied kubeconfig"
else
    echo "❌ Failed to copy kubeconfig or file is empty"
    exit 1
fi

# Update kubeconfig for external access
echo ""
echo "=== CONFIGURING KUBECTL FOR EXTERNAL ACCESS ==="
echo "Updating kubeconfig to use external IP address..."

# Replace internal IP addresses with external IP
sed -i.backup "s/127.0.0.1:6443/$AWS_INSTANCE_IP:6443/g" ~/.kube/config
sed -i.backup2 "s/$AWS_INSTANCE_PRIVATE_IP:6443/$AWS_INSTANCE_IP:6443/g" ~/.kube/config

echo "Updated server endpoint to use external IP: $AWS_INSTANCE_IP:6443"

# Configure TLS settings for external access
echo "Setting insecure-skip-tls-verify for external cluster access..."
kubectl config set-cluster cluster.local --server=https://$AWS_INSTANCE_IP:6443 --insecure-skip-tls-verify=true

echo "✅ Configured kubectl for external cluster access"

# Test kubectl connection
echo ""
echo "=== TESTING KUBECTL CONNECTION ==="
echo "Testing cluster connection..."
kubectl get nodes

if [ $? -eq 0 ]; then
    echo "✅ kubectl successfully connected to cluster!"
    echo ""
    echo "Cluster details:"
    kubectl get nodes -o wide
    echo ""
    echo "Cluster info:"
    kubectl cluster-info
else
    echo "❌ kubectl connection failed"
    echo ""
    echo "Required: Add port 6443 to your AWS security group"
    echo "• Type: Custom TCP, Port: 6443, Source: $(curl -s -4 ifconfig.me)/32"
    echo ""
    echo "Test connection manually:"
    echo "kubectl get nodes --insecure-skip-tls-verify=true"
fi
