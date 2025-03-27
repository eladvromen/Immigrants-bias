#!/usr/bin/env python
import os
import subprocess
import argparse

def run_experiment(model_name, run_type, input_file, output_dir):
    """Run a single experiment with the specified parameters"""
    cmd = [
        "python", "template_script.py",
        model_name,
        run_type,
        "--input", input_file,
        "--output", output_dir
    ]
    
    print(f"\n\n{'='*80}")
    print(f"Running experiment: {model_name} - {run_type}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    subprocess.run(cmd)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--input', default='data/t_and_a_immigration_dataset_curated.csv', 
                        help='Path to input dataset')
    
    args = parser.parse_args()
    
    # Define models and run types
    models = [
        "bert-base-uncased",
        "nlpaueb/legal-bert-base-uncased"
    ]
    
    run_types = [
        "translated",
        "legal",
        "legal_with_translated_concept"
    ]
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run all experiments
    for model in models:
        model_short_name = model.split('/')[-1]
        
        for run_type in run_types:
            # Create specific output directory for this experiment
            output_dir = f"results/{run_type}_test"
            os.makedirs(output_dir, exist_ok=True)
            
            # Run the experiment
            run_experiment(model, run_type, args.input, output_dir)

if __name__ == "__main__":
    main() 