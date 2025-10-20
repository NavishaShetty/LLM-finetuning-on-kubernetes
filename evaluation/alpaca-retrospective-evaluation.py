#!/usr/bin/env python3
"""
Correct Alpaca Evaluation - Matches Production Setup
Key fix: Does NOT use merge_and_unload() - keeps adapters active like production
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
MAX_LENGTH = 150  # Matching production default
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200

RESULTS_DIR = "./alpaca_evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("CORRECT ALPACA EVALUATION")
print("Matching production model loading (no merge_and_unload)")
print("=" * 80)

# Load models EXACTLY like production
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded")

print("\n[2/5] Loading BASE model for baseline...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
base_model.eval()
print("Base model loaded")

print("\n[3/5] Loading FINE-TUNED model (production setup)...")
# Load base model for fine-tuned version
finetuned_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

# Apply PEFT adapters - DO NOT MERGE (like production)
finetuned_model = PeftModel.from_pretrained(finetuned_base, FINETUNED_MODEL)
finetuned_model.eval()
print("Fine-tuned model loaded (adapters active, NOT merged)")
print(f"   Model type: {type(finetuned_model)}")

# Load dataset
print("\n[4/5] Loading Alpaca dataset...")
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
test_samples = alpaca.shuffle(seed=42).select(range(NUM_SAMPLES))
print(f"Selected {len(test_samples)} samples")

# Load metrics
print("\n[5/5] Loading evaluation metrics...")
rouge = load("rouge")
bleu = load("bleu")
meteor = load("meteor")
print("Metrics loaded")

def generate_response(model, prompt, is_finetuned=False):
    """
    Generate response EXACTLY like production API
    """
    # Format prompt for instruction-following (Alpaca format) - like production
    if is_finetuned:
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    else:
        # Base model doesn't need instruction formatting
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
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
    
    # Extract only the response part - like production
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

for i, sample in enumerate(tqdm(test_samples, desc="Generating"), 1):
    instruction = sample["instruction"]
    input_text = sample.get("input", "")
    reference = sample["output"]
    
    # Create the instruction prompt
    if input_text:
        full_instruction = f"{instruction}\n\nInput: {input_text}"
    else:
        full_instruction = instruction
    
    # Skip invalid samples
    if not reference or len(reference.strip()) < 5:
        continue
    
    try:
        # Generate from base model (no instruction format)
        base_pred = generate_response(base_model, full_instruction, is_finetuned=False)
        
        # Generate from fine-tuned model (with instruction format, like production)
        ft_pred = generate_response(finetuned_model, full_instruction, is_finetuned=True)
        
        # Skip if generation failed
        if len(base_pred.strip()) < 3 or len(ft_pred.strip()) < 3:
            continue
        
        base_predictions.append(base_pred)
        finetuned_predictions.append(ft_pred)
        references.append(reference)
        
        # Save examples
        if len(examples) < 20:
            examples.append({
                "instruction": instruction,
                "input": input_text if input_text else "(none)",
                "reference": reference,
                "base_prediction": base_pred,
                "finetuned_prediction": ft_pred,
                "base_length": len(base_pred.split()),
                "finetuned_length": len(ft_pred.split()),
                "reference_length": len(reference.split())
            })
    except Exception as e:
        print(f"\n    Warning: Skipped sample {i} due to error: {e}")
        continue

print(f"\nGenerated {len(base_predictions)} valid prediction pairs")

# Compute metrics
print("\n" + "=" * 80)
print("COMPUTING METRICS")
print("=" * 80)

print("  Computing ROUGE scores...")
base_rouge = rouge.compute(predictions=base_predictions, references=references)
ft_rouge = rouge.compute(predictions=finetuned_predictions, references=references)

print("  Computing BLEU scores...")
base_bleu = bleu.compute(predictions=base_predictions, references=[[r] for r in references])
ft_bleu = bleu.compute(predictions=finetuned_predictions, references=[[r] for r in references])

print("  Computing METEOR scores...")
base_meteor = meteor.compute(predictions=base_predictions, references=references)
ft_meteor = meteor.compute(predictions=finetuned_predictions, references=references)

print("All metrics computed")

# Compile results
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

# Calculate average lengths
avg_base_len = sum(len(p.split()) for p in base_predictions) / len(base_predictions)
avg_ft_len = sum(len(p.split()) for p in finetuned_predictions) / len(finetuned_predictions)
avg_ref_len = sum(len(r.split()) for r in references) / len(references)

# Save results
with open(f"{RESULTS_DIR}/results.json", "w") as f:
    json.dump({
        "dataset": "Alpaca",
        "num_samples": len(base_predictions),
        "model_loading": "PeftModel with active adapters (no merge)",
        "results": results,
        "improvements": improvements,
        "avg_lengths": {
            "base": avg_base_len,
            "finetuned": avg_ft_len,
            "reference": avg_ref_len
        }
    }, f, indent=2)

with open(f"{RESULTS_DIR}/examples.json", "w") as f:
    json.dump(examples, f, indent=2)

# Create comparison table
comparison_df = pd.DataFrame({
    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR"],
    "Base Model": [
        f"{results['base_model']['rouge1']:.4f}",
        f"{results['base_model']['rouge2']:.4f}",
        f"{results['base_model']['rougeL']:.4f}",
        f"{results['base_model']['bleu']:.4f}",
        f"{results['base_model']['meteor']:.4f}"
    ],
    "Fine-tuned": [
        f"{results['finetuned_model']['rouge1']:.4f}",
        f"{results['finetuned_model']['rouge2']:.4f}",
        f"{results['finetuned_model']['rougeL']:.4f}",
        f"{results['finetuned_model']['bleu']:.4f}",
        f"{results['finetuned_model']['meteor']:.4f}"
    ],
    "Improvement": [
        f"{improvements['rouge1']:+.1f}%",
        f"{improvements['rouge2']:+.1f}%",
        f"{improvements['rougeL']:+.1f}%",
        f"{improvements['bleu']:+.1f}%",
        f"{improvements['meteor']:+.1f}%"
    ]
})

comparison_df.to_csv(f"{RESULTS_DIR}/comparison.csv", index=False)

# Print results
print("\n" + "=" * 80)
print("EVALUATION RESULTS - CORRECT LOADING")
print("=" * 80)
print("\n" + comparison_df.to_string(index=False))

print("\n" + "=" * 80)
print("LENGTH ANALYSIS")
print("=" * 80)
print(f"Average response lengths:")
print(f"  Reference:   {avg_ref_len:.1f} words")
print(f"  Base model:  {avg_base_len:.1f} words")
print(f"  Fine-tuned:  {avg_ft_len:.1f} words")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"Evaluated on {len(base_predictions)} Alpaca samples")
print(f"ROUGE-L improvement: {improvements['rougeL']:+.1f}%")
print(f"BLEU improvement: {improvements['bleu']:+.1f}%")
print(f"METEOR improvement: {improvements['meteor']:+.1f}%")

avg_improvement = (improvements['rougeL'] + improvements['bleu'] + improvements['meteor']) / 3

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {RESULTS_DIR}/")
print(f"\nIMPORTANT: Check {RESULTS_DIR}/examples.json to see actual outputs")
print("Compare the quality, not just the scores!")