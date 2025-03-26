import pandas as pd
import numpy as np
import os

# Get paths to results
base_dir = 'results'
base_test_dir = os.path.join(base_dir, 'base_test')
legal_test_dir = os.path.join(base_dir, 'legal_test')

# Read CSV files (not TSV) based on the script's output format
translated_bert_df = pd.read_csv(os.path.join(base_test_dir, 'results_translated_bert-base-uncased.csv'))
translated_legal_bert_df = pd.read_csv(os.path.join(base_test_dir, 'results_translated_legal-bert-base-uncased.csv'))
legal_bert_df = pd.read_csv(os.path.join(legal_test_dir, 'results_legal_bert-base-uncased.csv'))
legal_legal_bert_df = pd.read_csv(os.path.join(legal_test_dir, 'results_legal_legal-bert-base-uncased.csv'))

# Print column names to verify
print("Column names in translated_bert_df:", translated_bert_df.columns.tolist())

def analyze_bias_results(df, model_name):
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Rename columns
    df_copy = df_copy.rename(columns={
        'concept1': 'negative_concept',
        'concept2': 'positive_concept',
        'score_sentence1': 'score_negative',
        'score_sentence2': 'score_positive'
    })
    
    # Group by category and subcategory for detailed analysis
    group_analysis = df_copy.groupby(['category', 'subcategory']).agg({
        'score_negative': ['mean', 'std', 'count'],
        'negative_concept': lambda x: x.iloc[0]  # Just take the first example
    }).round(4)
    
    # Calculate negative bias probability by group (how often negative concept is more probable)
    negative_bias = df_copy.groupby('group').apply(
        lambda x: (x['score_negative'] > x['score_positive']).mean()
    ).round(4)
    
    # Additional: Calculate average bias magnitude by group
    bias_magnitude = df_copy.groupby('group').apply(
        lambda x: (x['score_negative'] - x['score_positive']).mean()
    ).round(4)
    
    return {
        'model': model_name,
        'detailed_analysis': group_analysis,
        'negative_bias_by_group': negative_bias,
        'bias_magnitude_by_group': bias_magnitude
    }

# Analyze each dataset
results = {
    'translated_bert': analyze_bias_results(translated_bert_df, 'BERT (Translated)'),
    'translated_legal_bert': analyze_bias_results(translated_legal_bert_df, 'Legal BERT (Translated)'),
    'legal_bert': analyze_bias_results(legal_bert_df, 'BERT (Legal)'),
    'legal_legal_bert': analyze_bias_results(legal_legal_bert_df, 'Legal BERT (Legal)')
}

# Print results for each model
for model_name, result in results.items():
    print(f"\n=== Results for {result['model']} ===")
    print("\nNegative Bias Probability by Group:")
    print(result['negative_bias_by_group'])
    print("\nBias Magnitude by Group (negative - positive score):")
    print(result['bias_magnitude_by_group'])
    print("\nDetailed Analysis by Category/Subcategory:")
    print(result['detailed_analysis'])

# Optional: Combine all results for comparison
import matplotlib.pyplot as plt

# Compare negative bias probability across models
def plot_bias_comparison():
    # Extract bias data
    bias_data = {
        model_name: result['negative_bias_by_group'] 
        for model_name, result in results.items()
    }
    
    # Convert to DataFrame for plotting
    bias_df = pd.DataFrame(bias_data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bias_df.plot(kind='bar', ax=plt.gca())
    plt.title('Negative Bias Probability by Group Across Models')
    plt.ylabel('Probability of Negative Association')
    plt.xlabel('Group')
    plt.tight_layout()
    plt.savefig('bias_comparison.png')
    print("Bias comparison plot saved as 'bias_comparison.png'")

# Plot if matplotlib is available
try:
    plot_bias_comparison()
except:
    print("Could not create plot. Matplotlib may not be installed.")