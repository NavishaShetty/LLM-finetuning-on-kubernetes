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
- **Production Kubernetes Setup**: Deploy on AWS using Kubespray with proper GPU support (NVIDIA device plugin, RuntimeClass)
- **Concurrent Operations**: Run fine-tuning jobs alongside live inference services on the same GPU
- **Containerized Workflows**: Fully Dockerized training and inference pipelines
- **A/B Testing Ready**: Compare base model vs fine-tuned model performance side-by-side
- **Reproducible Pipeline**: Version-controlled configurations, datasets, and hyperparameters

## Architecture

Transform a base language model into a conversational AI through Kubernetes-orchestrated training:

Base TinyLlama → [K8s Training Job + Alpaca Dataset + QLoRA] → Fine-tuned Chat Model → Production API

**Infrastructure Stack:**
- **Compute**: AWS G4DN instance (Tesla T4 GPU, 32GB RAM)
- **Orchestration**: Kubernetes (deployed via Kubespray)
- **Training Framework**: PyTorch + HuggingFace Transformers + PEFT
- **Fine-Tuning Method**: QLoRA (4-bit quantization)
- **Dataset**: Stanford Alpaca (52K instruction-following examples)
- **Deployment**: FastAPI backend + Nginx frontend
- **Container Registry**: GitHub Container Registry (ghcr.io)

## Why This Project?

This project bridges the gap between ML research and production engineering:

1. **For ML Engineers**: Learn to operationalize LLM fine-tuning at scale
2. **For DevOps/Platform Engineers**: Understand ML workload requirements on Kubernetes
3. **For Students**: Hands-on experience with modern MLOps practices
4. **For Practitioners**: Production-ready template for custom LLM deployment

Unlike toy examples, this demonstrates:
- Real GPU resource management
- Persistent storage for model artifacts
- Job-based training workflows
- Live service updates without downtime
- Cost-effective cloud infrastructure

## Learning

- Setting up GPU-enabled Kubernetes clusters on AWS
- Configuring NVIDIA Container Toolkit and device plugins
- Implementing QLoRA for memory-efficient fine-tuning
- Creating Kubernetes Jobs for ML training workloads
- Managing model artifacts with PersistentVolumes
- Deploying and versioning ML models in production
- Building FastAPI inference services
- Monitoring GPU utilization in multi-tenant environments

**Training Specs:**
- Training Time: ~3-4 hours on Tesla T4
- GPU Memory Usage: ~6-8GB (training) + ~2GB (concurrent inference)
- Dataset: 52,000 instruction-response pairs
- Model Size: Base model 2.2GB + Adapters ~100MB

**Performance Improvement:**
- ✅ Better instruction following
- ✅ More coherent conversational responses
- ✅ Improved task completion
- ✅ Reduced hallucination on structured tasks

## Quick Start
```bash
# 1. Set up Kubernetes cluster with GPU support
./infrastructure/complete-gpu-setup.sh

# 2. Build and push training image
./scripts/push-training-image.sh

# 3. Submit training job
kubectl apply -f k8s-manifests/training-job.yaml

# 4. Monitor training
kubectl logs -f job/tinyllama-finetune-alpaca

# 5. Deploy fine-tuned model
kubectl apply -f k8s-manifests/inference/deployment-finetuned.yaml
