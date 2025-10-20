#!/usr/bin/env python3
"""
Master script to run all evaluations
Runs: Alpaca, Dolly-15k, and OpenOrca evaluations
"""

import subprocess
import sys
import os
from datetime import datetime
import json
import pandas as pd

print("=" * 80)
print("COMPREHENSIVE EVALUATION SUITE")
print("=" * 80)
print("This script will evaluate your fine-tuned model on THREE datasets:")
print("  1. Alpaca (in-distribution - same dataset used for training)")
print("  2. Dolly-15k (out-of-distribution)")
print("  3. OpenOrca (out-of-distribution - GPT-4 quality)")
print("\nEstimated total time: 45-60 minutes")
print("=" * 80)

# Create master results directory
MASTER_DIR = "./comprehensive_evaluation"
os.makedirs(MASTER_DIR, exist_ok=True)

start_time = datetime.now()
print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Track results
completed = []
failed = []

# Evaluation 1: Alpaca
print("\n" + "=" * 80)
print("EVALUATION 1/3: ALPACA")
print("=" * 80)
try:
    result = subprocess.run(
        [sys.executable, "alpaca-retrospective-evaluation.py"],
        capture_output=False,
        text=True,
        check=True
    )
    completed.append("Alpaca")
    print("\nAlpaca evaluation completed!")
except subprocess.CalledProcessError as e:
    failed.append("Alpaca")
    print(f"\nAlpaca evaluation failed: {e}")
except FileNotFoundError:
    failed.append("Alpaca")
    print("\nalpaca-retrospective-evaluation.py not found")

# Evaluation 2: Dolly-15k
print("\n" + "=" * 80)
print("EVALUATION 2/3: DOLLY-15K")
print("=" * 80)
try:
    result = subprocess.run(
        [sys.executable, "dolly-evaluation.py"],
        capture_output=False,
        text=True,
        check=True
    )
    completed.append("Dolly-15k")
    print("\nDolly-15k evaluation completed!")
except subprocess.CalledProcessError as e:
    failed.append("Dolly-15k")
    print(f"\nDolly-15k evaluation failed: {e}")
except FileNotFoundError:
    failed.append("Dolly-15k")
    print("\ndolly-evaluation.py not found")

# Evaluation 3: OpenOrca
print("\n" + "=" * 80)
print("EVALUATION 3/3: OPENORCA")
print("=" * 80)
try:
    result = subprocess.run(
        [sys.executable, "orca-evaluation.py"],
        capture_output=False,
        text=True,
        check=True
    )
    completed.append("OpenOrca")
    print("\nOpenOrca evaluation completed!")
except subprocess.CalledProcessError as e:
    failed.append("OpenOrca")
    print(f"\nOpenOrca evaluation failed: {e}")
except FileNotFoundError:
    failed.append("OpenOrca")
    print("\norca-evaluation.py not found")

# Compile results
print("\n" + "=" * 80)
print("COMPILING COMPREHENSIVE RESULTS")
print("=" * 80)

all_results = {}
dataset_dirs = {
    "Alpaca": "./alpaca_evaluation",
    "Dolly-15k": "./dolly_evaluation",
    "OpenOrca": "./orca_evaluation"
}

for dataset_name, dir_path in dataset_dirs.items():
    if dataset_name in completed:
        results_file = os.path.join(dir_path, "results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results[dataset_name] = json.load(f)

# Create comprehensive comparison
if all_results:
    print("\nCreating comprehensive comparison table...")
    
    comparison_data = []
    for dataset_name, data in all_results.items():
        comparison_data.append({
            "Dataset": dataset_name,
            "Type": "In-dist" if dataset_name == "Alpaca" else "Out-of-dist",
            "Samples": data.get("num_samples", "N/A"),
            "METEOR": f"{data['results']['finetuned_model']['meteor']:.4f}",
            "METEOR Δ": f"{data['improvements']['meteor']:+.1f}%",
            "ROUGE-L": f"{data['results']['finetuned_model']['rougeL']:.4f}",
            "ROUGE-L Δ": f"{data['improvements']['rougeL']:+.1f}%",
            "BLEU": f"{data['results']['finetuned_model']['bleu']:.4f}",
            "BLEU Δ": f"{data['improvements']['bleu']:+.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f"{MASTER_DIR}/all_datasets_comparison.csv", index=False)
    
    # Calculate average improvements (all datasets)
    avg_improvements = {
        "meteor": sum(d['improvements']['meteor'] for d in all_results.values()) / len(all_results),
        "rouge1": sum(d['improvements']['rouge1'] for d in all_results.values()) / len(all_results),
        "rougeL": sum(d['improvements']['rougeL'] for d in all_results.values()) / len(all_results),
        "bleu": sum(d['improvements']['bleu'] for d in all_results.values()) / len(all_results)
    }
    
    # Calculate out-of-distribution average
    ood_datasets = {k: v for k, v in all_results.items() if k != "Alpaca"}
    if ood_datasets:
        avg_ood_improvements = {
            "meteor": sum(d['improvements']['meteor'] for d in ood_datasets.values()) / len(ood_datasets),
            "rougeL": sum(d['improvements']['rougeL'] for d in ood_datasets.values()) / len(ood_datasets),
            "bleu": sum(d['improvements']['bleu'] for d in ood_datasets.values()) / len(ood_datasets)
        }
    else:
        avg_ood_improvements = None
    
    # Save comprehensive summary
    summary = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets_evaluated": completed,
        "datasets_failed": failed,
        "total_samples": sum(d.get("num_samples", 0) for d in all_results.values()),
        "average_improvements_all": avg_improvements,
        "average_improvements_ood": avg_ood_improvements,
        "individual_results": all_results
    }
    
    with open(f"{MASTER_DIR}/comprehensive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Results compiled")

# Print final summary
end_time = datetime.now()
duration = end_time - start_time

print("\n" + "=" * 80)
print("COMPREHENSIVE EVALUATION COMPLETE")
print("=" * 80)
print(f"\nStarted:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: {duration}")

print("\nEvaluation Summary:")
print(f"   Completed: {len(completed)} dataset(s)")
for dataset in completed:
    print(f"      • {dataset}")

if failed:
    print(f"\n   Failed: {len(failed)} dataset(s)")
    for dataset in failed:
        print(f"      • {dataset}")

if all_results:
    print("\n" + "=" * 80)
    print("CROSS-DATASET COMPARISON")
    print("=" * 80)
    print("\n" + comparison_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("AVERAGE IMPROVEMENTS - ALL DATASETS")
    print("=" * 80)
    print(f"   METEOR:   {avg_improvements['meteor']:+.1f}%")
    print(f"   ROUGE-L:  {avg_improvements['rougeL']:+.1f}%")
    print(f"   BLEU:     {avg_improvements['bleu']:+.1f}%")
    
    if avg_ood_improvements:
        print("\n" + "=" * 80)
        print("OUT-OF-DISTRIBUTION PERFORMANCE")
        print("=" * 80)
        print("(Excludes Alpaca - only unseen datasets)")
        print(f"\n   METEOR:   {avg_ood_improvements['meteor']:+.1f}%")
        print(f"   ROUGE-L:  {avg_ood_improvements['rougeL']:+.1f}%")
        print(f"   BLEU:     {avg_ood_improvements['bleu']:+.1f}%")

print("\n" + "=" * 80)
print("All evaluations complete!")
print("=" * 80)