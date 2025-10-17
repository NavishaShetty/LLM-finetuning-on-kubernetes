# Fine-Tuning LLM with Kubernetes

An end-to-end production-grade pipeline for fine-tuning Large Language Models (LLMs) 
using Kubernetes orchestration, demonstrating modern MLOps practices with GPU infrastructure.

## Project Overview

This project showcases how to fine-tune TinyLlama (1.1B parameters) into an instruction-following 
chat model using the Alpaca dataset, all orchestrated through Kubernetes on AWS GPU instances. 
The pipeline demonstrates memory-efficient fine-tuning with QLoRA (Quantized Low-Rank Adaptation) 
while keeping inference services running concurrently.

## Key Features

- **GPU-Accelerated Training**: Leverage NVIDIA Tesla T4 GPUs through Kubernetes for efficient model training
- **Memory-Efficient Fine-Tuning**: Implement QLoRA (4-bit quantization + LoRA adapters) to reduce GPU memory requirements by 75%
- **Production Infrastructure**: Deployed Kubernetes cluster on AWS using Kubespray with proper GPU support (NVIDIA device plugin, RuntimeClass)
- **Concurrent Operations**: Run fine-tuning jobs alongside live inference services on the same GPU
- **Containerized Workflows**: Fully Dockerized training and inference pipelines
- **Checkpoint Recovery**: Resume training after interruptions using AWS EBS
- **A/B Testing Ready**: Compare base model vs fine-tuned model performance side-by-side
- **GitOps Workflow**: Version-controlled infrastructure and configurations, datasets, and hyperparameters

## Architecture

Transform a base language model into a conversational AI through Kubernetes-orchestrated training:


AWS G4DN Instance (Tesla T4 GPU, 32GB RAM)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                      │
                        Kubernetes Cluster (Kubespray)
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │   TRAINING    │        │  BASE MODEL   │        │  FINE-TUNED   │
    │     JOB       │        │   INFERENCE   │        │   INFERENCE   │
    ├───────────────┤        ├───────────────┤        ├───────────────┤
    │ • QLoRA       │        │ • TinyLlama   │        │ • TinyLlama   │
    │ • GPU: 1      │        │ • GPU: 1      │        │   + LoRA      │
    │ • Mem: 8Gi    │        │ • Mem: 2Gi    │        │ • GPU: 1      │
    │ • Alpaca      │        │ • FastAPI     │        │ • Mem: 2Gi    │
    │   Dataset     │        │               │        │ • FastAPI     │
    └───────┬───────┘        └───────────────┘        └───────────────┘
            │                         │                         │
            │                         └────────┬────────────────┘
            │                                  │
            ▼                                  ▼
    ┌───────────────┐                ┌────────────────┐
    │ PERSISTENT    │                │  NVIDIA GPU    │
    │   VOLUME      │                │ DEVICE PLUGIN  │
    ├───────────────┤                └────────────────┘
    │ • EBS 50Gi    │                         │
    │ • Checkpoints │                         │
    │ • Artifacts   │        ┌────────────────┴────────────────┐
    └───────┬───────┘        │                                 │
            │                ▼                                 ▼
            │        ┌───────────────┐                ┌───────────────┐
            │        │   WEB UI      │                │  USER BROWSER │
            │        │  (Nginx)      │◄───────────────┤               │
            │        └───────────────┘                └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ HUGGING FACE  │
    │     HUB       │
    │ (Model Store) │
    └───────────────┘

External Services:
  🤗 Hugging Face Hub ◄──► Training Job (push/pull models)
  🌐 User Browser ────────► Web UI (Nginx) ──► APIs

***Training Pipeline***: TinyLlama-1.1B → [QLoRA + Alpaca Dataset] → Fine-tuned Model → HuggingFace Hub

### Technology Stack

| Component | Technology |
|-----------|------------|
| **Compute** | AWS G4DN (Tesla T4 GPU) |
| **Orchestration** | Kubernetes 1.28+ (Kubespray) |
| **Container Runtime** | containerd with NVIDIA runtime |
| **ML Framework** | PyTorch 2.0+, Transformers 4.35+ |
| **Fine-Tuning** | PEFT (QLoRA), 4-bit quantization |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML/CSS/JS, Nginx |
| **Storage** | AWS EBS (gp3) |
| **Registry** | GitHub Container Registry |

## Repository Structure

├── infrastructure/                # Kubernetes cluster setup scripts
├── training/                      # Fine-tuning QLoRA training script and Dockerfile
├── k8s-manifests/                 # Kubernetes resource definitions
│   ├── training-instruction-fintune/
│   │   ├── models-pv.yaml         # PersistentVolume
│   │   └── training-job.yaml      # Training Job
│   ├── inference-base/
│   │   ├── deployment.yaml        # Base model deployment
│   │   └── service.yaml           # Base model service
│   ├── inference-finetuned/
│   │   ├── deployment.yaml        # Fine-tuned deployment
│   │   └── service.yaml           # Fine-tuned service
│   ├── ui-base/
│   │   ├── configmap.yaml         # Base UI HTML
│   │   └── deployment.yaml        # Base UI deployment
│   └── ui-finetuned/
│       ├── configmap.yaml         # Fine-tuned UI HTML
│       └── deployment.yaml        # Fine-tuned UI deployment
├── inference-base-model/          # Base model FastAPI application
├── inference-finetuned-model/     # Fine-tuned model FastAPI application
└── docs/                          # Detailed documentation

## Quick Start

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/finetuning-llm-with-k8s.git
cd finetuning-llm-with-k8s
```

### Step 2: Configure AWS Connection
Edit `infrastructure/setup_aws_node.sh` with your instance details:

```bash
PUBLIC_IP="YOUR_AWS_PUBLIC_IP"       # e.g., "54.123.45.67"
PRIVATE_IP="YOUR_AWS_PRIVATE_IP"     # e.g., "172.31.0.10"
SSH_KEY_PATH="~/.ssh/your-key.pem"   # Path to your SSH key
SSH_USER="ubuntu"                     # SSH username
```

### Step 3: Deploy Kubernetes Cluster
```bash
# Make scripts executable
chmod +x infrastructure/*.sh

# Run automated setup (takes ~30-45 minutes)
./infrastructure/setup_aws_node.sh
```

This script will:
1. Install Kubespray and dependencies
2. Deploy Kubernetes to your AWS instance
3. Configure GPU support (NVIDIA device plugin)
4. Set up kubectl on your local machine
5. Verify cluster health

### Step 4: Create Persistent Storage
```bash
# Create directory on AWS instance
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_PUBLIC_IP "sudo mkdir -p /mnt/fast-disks/models && sudo chmod 777 /mnt/fast-disks/models"

# Deploy PersistentVolume
kubectl apply -f k8s-manifests/training-instruction-fintune/models-pv.yaml

# Verify
kubectl get pv
```

### Step 5: Configure Hugging Face Credentials
```bash
# Get your token from https://huggingface.co/settings/tokens
kubectl create secret generic huggingface-token \
  --from-literal=token='hf_YOUR_ACTUAL_TOKEN' \
  --from-literal=repo='your-username/tinyllama-alpaca-finetuned'
```

### Step 6: Launch Training Job
```bash
# Deploy training job
kubectl apply -f k8s-manifests/training-instruction-fintune/training-job.yaml

# Monitor progress
kubectl logs -f job/tinyllama-finetune-alpaca
```

### Step 7: Deploy Inference Services

#### Deploy Base Model
```bash
kubectl apply -f k8s-manifests/inference-base/
```

#### Deploy Fine-Tuned Model (after training completes)
```bash
kubectl apply -f k8s-manifests/inference-finetuned/
```

#### Deploy Web UIs
```bash
# Base model UI
kubectl apply -f k8s-manifests/ui-base/

# Fine-tuned model UI
kubectl apply -f k8s-manifests/ui-finetuned/
```

### Step 8: Access Services

```bash
# Get NodePort for services
kubectl get svc

# Example output:
# llm-api-finetuned-service   NodePort   10.96.1.1   <none>   80:30557/TCP
# llm-ui-finetuned-service    NodePort   10.96.1.2   <none>   80:31234/TCP
```

Access in browser:
- **Fine-tuned API**: `http://YOUR_PUBLIC_IP:30557`
- **Fine-tuned UI**: `http://YOUR_PUBLIC_IP:31234`
