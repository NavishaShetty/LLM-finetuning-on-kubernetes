#!/usr/bin/env python3
"""
Dolly-15k Evaluation - Matches Production Setup
Out-of-distribution evaluation (not seen during training)
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import json
import pandas as pd
from evaluate import load

# Configuration - EXACTLY matching production
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINETUNED_MODEL = "shettynavisha25/tinyllama-alpaca-finetuned"
MAX_LENGTH = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200

RESULTS_DIR = "./dolly_evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("DOLLY-15K EVALUATION (Out-of-Distribution)")
print("=" * 80)

# Load models
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded")

print("\n[2/5] Loading BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
base_model.eval()
print("Base model loaded")

print("\n[3/5] Loading FINE-TUNED model...")
finetuned_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
finetuned_model = PeftModel.from_pretrained(finetuned_base, FINETUNED_MODEL)
finetuned_model.eval()
print("Fine-tuned model loaded (adapters active)")

# Load dataset
print("\n[4/5] Loading Dolly-15k dataset...")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
test_samples = dolly.shuffle(seed=42).select(range(NUM_SAMPLES))
print(f"Loaded {len(test_samples)} samples")

# Load metrics
print("\n[5/5] Loading evaluation metrics...")
rouge = load("rouge")
bleu = load("bleu")
meteor = load("meteor")
print("Metrics loaded")

def generate_response(model, prompt, is_finetuned=False):
    """Generate response matching production setup"""
    if is_finetuned:
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in generated_text:
        generated_text = generated_text.split("### Response:")[-1].strip()
    
    return generated_text

# Generate predictions
print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)
print("This may take 10-15 minutes...\n")

base_predictions = []
finetuned_predictions = []
references = []
examples = []

for sample in tqdm(test_samples, desc="Generating"):
    instruction = sample["instruction"]
    context = sample.get("context", "")
    reference = sample["response"]
    
    # Create prompt
    if context:
        full_instruction = f"{instruction}\n\nContext: {context}"
    else:
        full_instruction = instruction
    
    if not reference or len(reference.strip()) < 5:
        continue
    
    try:
        base_pred = generate_response(base_model, full_instruction, is_finetuned=False)
        ft_pred = generate_response(finetuned_model, full_instruction, is_finetuned=True)
        
        if len(base_pred.strip()) < 3 or len(ft_pred.strip()) < 3:
            continue
        
        base_predictions.append(base_pred)
        finetuned_predictions.append(ft_pred)
        references.append(reference)
        
        if len(examples) < 20:
            examples.append({
                "instruction": instruction,
                "context": context if context else "(none)",
                "reference": reference,
                "base_prediction": base_pred,
                "finetuned_prediction": ft_pred,
                "base_length": len(base_pred.split()),
                "finetuned_length": len(ft_pred.split()),
                "reference_length": len(reference.split())
            })
    except Exception as e:
        continue

print(f"\nGenerated {len(base_predictions)} valid prediction pairs")

# Compute metrics
print("\n" + "=" * 80)
print("COMPUTING METRICS")
print("=" * 80)

base_rouge = rouge.compute(predictions=base_predictions, references=references)
ft_rouge = rouge.compute(predictions=finetuned_predictions, references=references)
base_bleu = bleu.compute(predictions=base_predictions, references=[[r] for r in references])
ft_bleu = bleu.compute(predictions=finetuned_predictions, references=[[r] for r in references])
base_meteor = meteor.compute(predictions=base_predictions, references=references)
ft_meteor = meteor.compute(predictions=finetuned_predictions, references=references)

results = {
    "base_model": {
        "rouge1": base_rouge["rouge1"],
        "rouge2": base_rouge["rouge2"],
        "rougeL": base_rouge["rougeL"],
        "bleu": base_bleu["bleu"],
        "meteor": base_meteor["meteor"]
    },
    "finetuned_model": {
        "rouge1": ft_rouge["rouge1"],
        "rouge2": ft_rouge["rouge2"],
        "rougeL": ft_rouge["rougeL"],
        "bleu": ft_bleu["bleu"],
        "meteor": ft_meteor["meteor"]
    }
}

improvements = {}
for metric in ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]:
    base_score = results["base_model"][metric]
    ft_score = results["finetuned_model"][metric]
    improvement = ((ft_score - base_score) / base_score * 100) if base_score > 0 else 0
    improvements[metric] = improvement

# Save results
with open(f"{RESULTS_DIR}/results.json", "w") as f:
    json.dump({
        "dataset": "Dolly-15k (Out-of-Distribution)",
        "num_samples": len(base_predictions),
        "results": results,
        "improvements": improvements
    }, f, indent=2)

with open(f"{RESULTS_DIR}/examples.json", "w") as f:
    json.dump(examples, f, indent=2)

comparison_df = pd.DataFrame({
    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR"],
    "Base Model": [f"{results['base_model'][m]:.4f}" for m in ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]],
    "Fine-tuned": [f"{results['finetuned_model'][m]:.4f}" for m in ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]],
    "Improvement": [f"{improvements[m]:+.1f}%" for m in ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]]
})

comparison_df.to_csv(f"{RESULTS_DIR}/comparison.csv", index=False)

# Print results
print("\n" + "=" * 80)
print("EVALUATION RESULTS - DOLLY-15K")
print("=" * 80)
print("\n" + comparison_df.to_string(index=False))

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"Evaluated on {len(base_predictions)} Dolly-15k samples (UNSEEN DATA)")
print(f"METEOR improvement: {improvements['meteor']:+.1f}%")
print(f"ROUGE-L improvement: {improvements['rougeL']:+.1f}%")
print(f"BLEU improvement: {improvements['bleu']:+.1f}%")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {RESULTS_DIR}/")