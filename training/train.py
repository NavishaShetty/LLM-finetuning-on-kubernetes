#!/usr/bin/env python3
"""
Fine-tune TinyLlama on Alpaca dataset using QLoRA
With checkpoint recovery and Hugging Face Hub integration
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import transformers
from datetime import datetime

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/models/tinyllama-alpaca-finetuned")
DATASET_NAME = "tatsu-lab/alpaca"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LOGGING_STEPS = 10
SAVE_STEPS = 500

# Hugging Face Hub configuration
HF_TOKEN = os.getenv("HF_TOKEN") 
HF_REPO = os.getenv("HF_REPO", "shettynavisha25/tinyllama-alpaca-finetuned")

print("=" * 80)
print("TinyLlama Fine-Tuning with QLoRA on Alpaca Dataset")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Output: {OUTPUT_DIR}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if HF_TOKEN:
    print(f"Hugging Face: Will push to {HF_REPO}")
else:
    print("Hugging Face: Push disabled (no HF_TOKEN)")
print("=" * 80)

# Check for existing checkpoints (for resume capability)
checkpoint = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        # Sort by checkpoint number and get the latest
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
        print(f"\nRESUMING FROM CHECKPOINT")
        print(f"   Found checkpoint: {checkpoint}")
        print(f"   Training will continue from this point...")
        print("=" * 80)

# Load tokenizer
print("\n[1/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded")

# Load model with 4-bit quantization
print("\n[2/6] Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded")

# Prepare model for training
print("\n[3/6] Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
print("Model prepared")

# Configure LoRA
print("\n[4/6] Configuring LoRA...")
lora_config = LoraConfig(
    r=16,                      # LoRA rank
    lora_alpha=32,             # LoRA alpha
    target_modules=[           # Which modules to apply LoRA to
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("LoRA configured")

# Load and prepare dataset
print("\n[5/6] Loading Alpaca dataset...")
dataset = load_dataset(DATASET_NAME)

def format_alpaca(sample):
    """Format Alpaca dataset samples into instruction format"""
    instruction = sample["instruction"]
    input_text = sample["input"]
    output = sample["output"]
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": prompt}

# Format dataset
dataset = dataset.map(format_alpaca, remove_columns=dataset["train"].column_names)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

print("Tokenizing dataset (this may take a few minutes)...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Use full training dataset
train_dataset = tokenized_dataset["train"]
print(f"Dataset loaded: {len(train_dataset)} training samples")

# Training arguments
print("\n[6/6] Setting up training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=LOGGING_STEPS,
    
    # CHECKPOINT SETTINGS (for crash recovery)
    save_strategy="steps",
    save_steps=SAVE_STEPS,              # Save every 500 steps (~40 min)
    save_total_limit=3,                 # Keep last 3 checkpoints on EBS
    load_best_model_at_end=False,       # Don't need best model for this task
    
    # PUSH TO HUB SETTINGS (for final model upload)
    push_to_hub=bool(HF_TOKEN),         # Only if token provided
    hub_model_id=HF_REPO if HF_TOKEN else None,
    hub_strategy="end",                 # Only push final model (not checkpoints)
    hub_token=HF_TOKEN,
    
    # Other settings
    gradient_checkpointing=True,        # Save memory
    report_to="none",                   # Disable wandb/tensorboard
    optim="paged_adamw_8bit",
    warmup_steps=100,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# Start training (with resume capability)
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Expected duration: ~11-13 hours on T4 GPU")
if checkpoint:
    print(f"Resuming from: {checkpoint}")
print("=" * 80 + "\n")

# Train with checkpoint resume
trainer.train(resume_from_checkpoint=checkpoint)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

# Save the final model locally
print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model saved locally!")

# Push to Hub happens automatically if push_to_hub=True
if HF_TOKEN:
    print(f"\nModel automatically pushed to Hugging Face Hub")
    print(f"View at: https://huggingface.co/{HF_REPO}")
else:
    print("\nTo push to Hugging Face, set HF_TOKEN environment variable")

print("\n" + "=" * 80)
print("Fine-tuning completed successfully!")
print(f"Model location (local): {OUTPUT_DIR}")
if HF_TOKEN:
    print(f"Model location (Hub): https://huggingface.co/{HF_REPO}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)