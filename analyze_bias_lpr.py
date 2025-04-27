import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def calculate_lpr(df):
    """Calculates the Log Probability Ratio (LPR) and adds it as a column."""
    # LPR = score_concept1 - score_concept2
    # Negative LPR indicates higher score for concept1 (negative concept)
    df['lpr'] = df['score_sentence1'] - df['score_sentence2']
    return df

def get_bias_category(lpr_value):
    """Categorizes bias strength based on LPR value."""
    abs_lpr = abs(lpr_value)
    if abs_lpr < 0.1:
        return "Minimal"
    elif 0.1 <= abs_lpr <= 0.3:
        return "Moderate"
    else:
        return "Strong"

def aggregate_results(df):
    """Aggregates LPR results by group."""
    grouped = df.groupby('group')['lpr']
    
    summary = grouped.agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate 95% confidence interval
    summary['ci_95_low'], summary['ci_95_high'] = st.t.interval(
        confidence=0.95, 
        df=summary['count']-1, 
        loc=summary['mean'], 
        scale=summary['std'] / np.sqrt(summary['count'])
        )
        
    # Handle cases with low counts where CI might be NaN
    summary.fillna({'ci_95_low': summary['mean'], 'ci_95_high': summary['mean']}, inplace=True)
    
    summary['bias_category'] = summary['mean'].apply(get_bias_category)
    
    return summary

def create_visualizations(df, summary, output_dir):
    """Creates and saves the visualizations."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='group', y='lpr')
    plt.title('LPR Distribution by Group')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lpr_boxplot.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    summary['error'] = summary['ci_95_high'] - summary['mean'] # Error bar length from mean
    plt.bar(summary['group'], summary['mean'], yerr=summary['error'], capsize=5)
    plt.axhline(0, color='grey', linestyle='--')
    plt.ylabel('Mean LPR (95% CI)')
    plt.title('Mean LPR by Group with 95% Confidence Intervals')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_lpr_barchart.png'))
    plt.close()

    # Heatmap of mean LPR across subcategory and group
    heatmap_data = df.groupby(['subcategory', 'group'])['lpr'].mean().unstack()
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title('Mean LPR Heatmap: Subcategory vs. Group')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lpr_heatmap.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def perform_t_tests(df, reference_group='people'):
    """Performs t-tests comparing each group to the reference group."""
    groups = df['group'].unique()
    results = {}
    
    if reference_group not in groups:
        print(f"Warning: Reference group '{reference_group}' not found in data. Skipping t-tests.")
        return None
        
    ref_data = df[df['group'] == reference_group]['lpr']
    
    print(f"--- T-Tests (vs. {reference_group}) ---")
    for group in groups:
        if group == reference_group:
            continue
        
        group_data = df[df['group'] == group]['lpr']
        
        # Check for sufficient data and variance
        if len(group_data) < 2 or len(ref_data) < 2 or np.var(group_data) == 0 or np.var(ref_data) == 0:
             print(f"Skipping T-test for {group}: Insufficient data or zero variance.")
             results[group] = {'t_stat': np.nan, 'p_value': np.nan, 'significant': False}
             continue
             
        t_stat, p_value = st.ttest_ind(group_data, ref_data, equal_var=False, nan_policy='omit') # Welch's t-test
        significant = p_value < 0.05
        results[group] = {'t_stat': t_stat, 'p_value': p_value, 'significant': significant}
        
        print(f"{group} vs {reference_group}: t = {t_stat:.3f}, p = {p_value:.3f} {'(Significant)' if significant else ''}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze immigration bias using Log Probability Ratios (LPR).')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('--ref_group', type=str, default='people', help='Reference group for t-tests (default: people).')
    parser.add_argument('--output_dir', type=str, default='analysis_output', help='Directory to save plots (default: analysis_output).')

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File not found at {args.csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Basic validation of required columns
    required_cols = ['group', 'score_sentence1', 'score_sentence2', 'subcategory', 'templateId']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain columns: {', '.join(required_cols)}")
        return

    df = calculate_lpr(df)
    summary_stats = aggregate_results(df)

    print("--- Aggregated LPR Statistics by Group ---")
    print(summary_stats.to_string(index=False, float_format="%.3f"))

    # Define output directory based on input filename
    base_output_dir = args.output_dir
    file_specific_output_dir = os.path.join(
        base_output_dir, 
        os.path.splitext(os.path.basename(args.csv_file))[0] + "_analysis"
    )
    
    create_visualizations(df, summary_stats, file_specific_output_dir)
    perform_t_tests(df, reference_group=args.ref_group)

if __name__ == "__main__":
    main() 