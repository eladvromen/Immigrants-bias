import os
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
import sys

# Set environment variables
os.environ["HF_HOME"] = "/data/resource/huggingface"

def replace_concept_in_sentence(row, sentence, run_type='translated'):
    """
    Replace concept placeholders in sentence based on run type
    run_type: 'translated' or 'legal'
    """
    if run_type == 'translated':
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

def sentence_score_with_group(row, sentence, sentence_template, group, tokenizer, model, results, stereo_score_dict, stereo_groups_dict):
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

    # Get sentence variations based on run type
    run_type = 'translated' if 'sentence_translated' in sentence else 'legal'
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

def sentence_score_without_group(row, sentence, sentence_template, gender, tokenizer, model, results, stereo_score):
    log_softmax = torch.nn.LogSoftmax(dim=1)
    print(sentence)

    # Get sentence variations based on run type
    run_type = 'translated' if 'sentence_translated' in sentence else 'legal'
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

def main(model_name, run_type):
    # Define file paths using relative paths
    input_file = os.path.join("data", "t_and_a_immigration_dataset.csv")
    
    # Set output directory based on run_type
    if run_type == 'translated':
        output_dir = os.path.join("results", "base_test")
    else:  # legal
        output_dir = os.path.join("results", "legal_test")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
        'concept1', 'concept2', 'score_sentence1', 'score_sentence2'
    ])

    # Select appropriate columns based on run type
    if run_type == 'translated':
        sentence_col = 'sentence_translated'
        group_col = 'group_translated'
    else:  # legal
        sentence_col = 'legal_augmentation'
        group_col = 'group_translated'

    # Verify columns exist
    if sentence_col not in dataset.columns:
        print(f"\nERROR: Column '{sentence_col}' not found!")
        print("Available columns:", dataset.columns.tolist())
        raise ValueError(f"Column '{sentence_col}' not found in dataset")
    
    if group_col not in dataset.columns:
        print(f"\nERROR: Column '{group_col}' not found!")
        print("Available columns:", dataset.columns.tolist())
        raise ValueError(f"Column '{group_col}' not found in dataset")

    # Process each row
    for i, row in dataset.iterrows():
        sentence_template = str(row[sentence_col])
        
        if '[GROUP]' in sentence_template:
            # Handle templates with [GROUP]
            groups = [g.strip() for g in str(row[group_col]).split(',')]
            
            for group in groups:
                if not group:  # Skip empty groups
                    continue
                    
                # Create sentence with current group
                sentence = sentence_template.replace('[GROUP]', group)
                
                results, stereo_score_dict, stereo_groups_dict = sentence_score_with_group(
                    row, sentence, sentence_template, group,
                    tokenizer, model, results, stereo_score_dict, stereo_groups_dict
                )
        else:
            # Handle templates without [GROUP]
            results, stereo_score = sentence_score_without_group(
                row, sentence_template, sentence_template, 'masc',  # gender default to 'masc' as per original script
                tokenizer, model, results, 0  # Initialize stereo_score as 0
            )

    # Calculate and print results
    model_short_name = model_name.split('/')[-1]
    
    # Print group-based results
    for group, count in stereo_score_dict.items():
        percentage = round((count / stereo_groups_dict[group]) * 100, 2)
        print(f'Bias score for "{group}" group: {percentage}%')

    # Save results in both formats
    tsv_filename = os.path.join(output_dir, f'results_{run_type}_{model_short_name}.tsv')
    csv_filename = os.path.join(output_dir, f'results_{run_type}_{model_short_name}.csv')
    
    results.to_csv(tsv_filename, sep='\t', index=False)
    results.to_csv(csv_filename, index=False)
    
    print(f"\nResults saved as:")
    print(f"- {tsv_filename} (TSV format)")
    print(f"- {csv_filename} (CSV format)")

if __name__ == "__main__":
    # Example usage:
    # python template_script.py bert-base-uncased translated
    # python template_script.py nlpaueb/legal-bert-base-uncased legal
    model_name = sys.argv[1]
    run_type = sys.argv[2]  # 'translated' or 'legal'
    main(model_name, run_type) 