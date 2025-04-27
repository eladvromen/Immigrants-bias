import os
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
import sys
import argparse
import datetime

# Set environment variables
os.environ["HF_HOME"] = "/data/resource/huggingface"

def replace_concept_in_sentence(row, sentence, run_type='translated'):
    """
    Replace concept placeholders in sentence based on run type
    run_type: 'translated', 'legal', or 'legal_with_translated_concept'
    """
    if run_type in ['translated', 'legal_with_translated_concept']:
        sentence1 = sentence.replace('[CONCEPT]', row['concept1_masc_translated'])
        sentence2 = sentence.replace('[CONCEPT]', row['concept2_masc_translated'])
        concept1 = row['concept1_masc_translated']
        concept2 = row['concept2_masc_translated']
    else:  # legal
        sentence1 = sentence.replace('[CONCEPT]', row['concept1_mac_legal'])
        sentence2 = sentence.replace('[CONCEPT]', row['concept2_mac_legal'])
        concept1 = row['concept1_mac_legal']
        concept2 = row['concept2_mac_legal']

    return sentence1, concept1, sentence2, concept2

def sentence_score_with_group(row, sentence, sentence_template, group, tokenizer, model, results, stereo_score_dict, stereo_groups_dict, run_type):
    log_softmax = torch.nn.LogSoftmax(dim=1)
    print(sentence)

    # Simplify group classification
    group_lower = group.lower().strip()
    if 'refugee' in group_lower:
        simplified_group = 'refugees'
    elif 'immigrant' in group_lower:
        simplified_group = 'immigrants'
    elif 'foreign' in group_lower:
        simplified_group = 'foreigner'
    elif 'people from other countries' in group_lower:
        simplified_group = 'people_from_other_countries'
    else:
        simplified_group = 'people'

    # Track group statistics
    stereo_groups_dict[simplified_group] = stereo_groups_dict.get(simplified_group, 0) + 1

    # Use the passed run_type instead of determining it from the sentence
    sentence1, concept1, sentence2, concept2 = replace_concept_in_sentence(row, sentence, run_type)

    # Process sentences through model
    sentence1_token_ids = tokenizer.encode(sentence1, return_tensors='pt')
    sentence2_token_ids = tokenizer.encode(sentence2, return_tensors='pt')
    
    with torch.no_grad():
        score_sentence1 = calculate_aul_for_bert(model, sentence1_token_ids, log_softmax)
        score_sentence2 = calculate_aul_for_bert(model, sentence2_token_ids, log_softmax)

    # Track stereotypical scores
    if score_sentence1 > score_sentence2:
        stereo_score_dict[simplified_group] = stereo_score_dict.get(simplified_group, 0) + 1
        most_probable_concept = concept1
    else:
        most_probable_concept = concept2

    # Create new row data
    new_row = pd.DataFrame([{
        'templateId': row['templateId'],
        'category': row['category'],
        'subcategory': row['subcategory'],
        'group': simplified_group,
        'sentence_template': sentence_template,
        'most_probable_concept': most_probable_concept,
        'concept1': concept1,
        'concept2': concept2,
        'score_sentence1': score_sentence1,
        'score_sentence2': score_sentence2
    }])

    # Use concat instead of append
    results = pd.concat([results, new_row], ignore_index=True)
    
    return results, stereo_score_dict, stereo_groups_dict

def calculate_aul_for_bert(model, token_ids, log_softmax):
    '''
    Calculate averaged log probability of unmasked sequence (AUL)
    '''
    output = model(token_ids)
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    log_prob = torch.mean(token_log_probs)
    return log_prob.item()

def sentence_score_without_group(row, sentence, sentence_template, gender, tokenizer, model, results, stereo_score, run_type):
    log_softmax = torch.nn.LogSoftmax(dim=1)
    print(sentence)

    # Use the passed run_type instead of determining it from the sentence
    sentence1, concept1, sentence2, concept2 = replace_concept_in_sentence(row, sentence, run_type)

    # Process sentences through model
    sentence1_token_ids = tokenizer.encode(sentence1, return_tensors='pt')
    sentence2_token_ids = tokenizer.encode(sentence2, return_tensors='pt')
    
    with torch.no_grad():
        score_sentence1 = calculate_aul_for_bert(model, sentence1_token_ids, log_softmax)
        score_sentence2 = calculate_aul_for_bert(model, sentence2_token_ids, log_softmax)

    if score_sentence1 > score_sentence2:
        stereo_score += 1
        most_probable_concept = concept1
    else:
        most_probable_concept = concept2

    # Create new row data
    new_row = pd.DataFrame([{
        'templateId': row['templateId'],
        'category': row['category'],
        'subcategory': row['subcategory'],
        'group': 'Null',  # Explicitly mark as Null for non-group templates
        'sentence_template': sentence_template,
        'most_probable_concept': most_probable_concept,
        'concept1': concept1,
        'concept2': concept2,
        'score_sentence1': score_sentence1,
        'score_sentence2': score_sentence2
    }])

    # Use concat instead of append
    results = pd.concat([results, new_row], ignore_index=True)

    return results, stereo_score

def get_top_completions(sentence_template, tokenizer, model, top_k=3):
    """
    Get the top k most probable completions for a [CONCEPT] placeholder
    """
    # Replace [CONCEPT] with [MASK] token
    masked_sentence = sentence_template.replace('[CONCEPT]', tokenizer.mask_token)
    
    # Encode the sentence
    inputs = tokenizer(masked_sentence, return_tensors='pt')
    
    # Get the position of the [MASK] token
    mask_position = torch.where(inputs['input_ids'][0] == tokenizer.mask_token_id)[0]
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits[0, mask_position]
        
    # Get top k predictions
    top_k_scores, top_k_tokens = torch.topk(predictions, top_k, dim=1)
    
    # Convert to words and probabilities
    completions = []
    for scores, tokens in zip(top_k_scores[0], top_k_tokens[0]):
        word = tokenizer.decode(tokens)
        prob = torch.softmax(predictions, dim=1)[0][tokens].item()
        completions.append((word.strip(), prob))
    
    return completions

def process_template(row, sentence_template, group, tokenizer, model, results, 
                    stereo_score_dict, stereo_groups_dict, run_type):
    """
    Process a single template, getting both paired concept scores and top completions
    """
    # Get unconditional top completions
    top_completions = get_top_completions(sentence_template, tokenizer, model)
    
    # Get the regular AUL scores for concept pairs
    if group:
        results, stereo_score_dict, stereo_groups_dict = sentence_score_with_group(
            row, sentence_template, sentence_template, group,
            tokenizer, model, results, stereo_score_dict, stereo_groups_dict, run_type
        )
    else:
        results, _ = sentence_score_without_group(
            row, sentence_template, sentence_template, 'masc',
            tokenizer, model, results, 0, run_type
        )
    
    # Add top completions to the last row of results
    last_row_idx = results.index[-1]
    for i, (word, prob) in enumerate(top_completions, 1):
        results.at[last_row_idx, f'top_{i}_completion'] = word
        results.at[last_row_idx, f'top_{i}_probability'] = prob
    
    return results, stereo_score_dict, stereo_groups_dict

def main(model_name, run_type, input_file=None, output_dir=None):
    # Define default file paths
    default_input = os.path.join("data", "t_and_a_immigration_dataset.csv")
    input_file = input_file or default_input
    
    # Set default output directory based on run_type
    default_output = os.path.join("results", f"{run_type}_test")
    output_dir = output_dir or default_output
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split('/')[-1]
    log_file = os.path.join(output_dir, f"log_{timestamp}_{run_type}_{model_short_name}.txt")
    
    # Redirect stdout to both console and log file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Save original stdout and redirect to logger
    original_stdout = sys.stdout
    sys.stdout = Logger(log_file)
    
    # Log experiment details
    print(f"=== Experiment Details ===")
    print(f"Model: {model_name}")
    print(f"Run type: {run_type}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==========================\n")
    
    # Load dataset with explicit handling of separators and cleaning
    try:
        # First attempt to read with tab separator
        dataset = pd.read_csv(input_file, sep='\t')
        
        # If we got only one column with commas, we need to split it
        if len(dataset.columns) == 1:
            # Read the file again treating it as comma-separated
            dataset = pd.read_csv(input_file, sep=',')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.stdout = original_stdout  # Restore original stdout
        return

    # Clean column names
    dataset.columns = (dataset.columns
                      .str.strip()                    # Remove leading/trailing whitespace
                      .str.replace(' +', ' ', regex=True)  # Replace multiple spaces with single space
                      .str.strip())                   # Remove any remaining whitespace
    
    # Print column names for debugging
    print("\nCleaned column names:")
    for col in dataset.columns:
        print(f"'{col}'")

    # Initialize model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize tracking dictionaries
    stereo_score_dict = {}
    stereo_groups_dict = {}
    
    # Initialize results DataFrame
    results = pd.DataFrame(columns=[
        'templateId', 'category', 'subcategory', 'group', 
        'sentence_template', 'most_probable_concept', 
        'concept1', 'concept2', 'score_sentence1', 'score_sentence2',
        'top_1_completion', 'top_1_probability',
        'top_2_completion', 'top_2_probability',
        'top_3_completion', 'top_3_probability'
    ])

    # Select appropriate columns based on run type
    if run_type == 'translated':
        sentence_col = 'sentence_translated'
        group_col = 'group_translated'
    else:  # legal or legal_with_translated_concept
        sentence_col = 'legal_augmentation'
        group_col = 'group_translated'

    # Verify columns exist
    if sentence_col not in dataset.columns:
        print(f"\nERROR: Column '{sentence_col}' not found!")
        print("Available columns:", dataset.columns.tolist())
        sys.stdout = original_stdout  # Restore original stdout
        raise ValueError(f"Column '{sentence_col}' not found in dataset")
    
    if group_col not in dataset.columns:
        print(f"\nERROR: Column '{group_col}' not found!")
        print("Available columns:", dataset.columns.tolist())
        sys.stdout = original_stdout  # Restore original stdout
        raise ValueError(f"Column '{group_col}' not found in dataset")

    # Process each row
    for i, row in dataset.iterrows():
        sentence_template = str(row[sentence_col])
        
        if '[GROUP]' in sentence_template:
            groups = [g.strip() for g in str(row[group_col]).split(',')]
            
            for group in groups:
                if not group:  # Skip empty groups
                    continue
                    
                sentence = sentence_template.replace('[GROUP]', group)
                results, stereo_score_dict, stereo_groups_dict = process_template(
                    row, sentence, group, tokenizer, model,
                    results, stereo_score_dict, stereo_groups_dict, run_type
                )
        else:
            results, stereo_score_dict, stereo_groups_dict = process_template(
                row, sentence_template, None, tokenizer, model,
                results, stereo_score_dict, stereo_groups_dict, run_type
            )

    # Calculate and print results
    model_short_name = model_name.split('/')[-1]
    
    # Print group-based results
    print("\n=== Bias Score Results ===")
    for group, count in stereo_score_dict.items():
        percentage = round((count / stereo_groups_dict[group]) * 100, 2)
        print(f'Bias score for "{group}" group: {percentage}%')
    print("==========================\n")

    # Save results in both formats
    tsv_filename = os.path.join(output_dir, f'results_{run_type}_{model_short_name}.tsv')
    csv_filename = os.path.join(output_dir, f'results_{run_type}_{model_short_name}.csv')
    
    results.to_csv(tsv_filename, sep='\t', index=False)
    results.to_csv(csv_filename, index=False)
    
    print(f"\nResults saved as:")
    print(f"- {tsv_filename} (TSV format)")
    print(f"- {csv_filename} (CSV format)")
    print(f"- Log file: {log_file}")
    
    # Restore original stdout
    sys.stdout = original_stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run bias analysis on transformer models')
    parser.add_argument('model_name', help='Name or path of the HuggingFace model')
    parser.add_argument('run_type', 
                       choices=['translated', 'legal', 'legal_with_translated_concept'], 
                       help='Type of run to perform')
    parser.add_argument('--input', help='Path to input dataset (CSV/TSV format)')
    parser.add_argument('--output', help='Path to output directory')
    
    args = parser.parse_args()
    main(args.model_name, args.run_type, args.input, args.output) 