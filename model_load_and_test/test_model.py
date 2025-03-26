from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import os

# Set environment variables
os.environ["HF_HOME"] = "/data/resource/huggingface"

# Path to the model-last directory
model_path = "/data/shil6369/legal_bert_project/models/asylex_model/pretrained_legalbert/model-last"

def load_immigration_model():
    try:
        # Load just the transformer component using the base model name
        model = AutoModelForMaskedLM.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
        # Load the fine-tuned weights with map_location
        transformer_weights_path = os.path.join(model_path, "transformer")
        if os.path.exists(transformer_weights_path):
            print(f"Found transformer weights at: {transformer_weights_path}")
            model_file = os.path.join(transformer_weights_path, "model")
            print(f"Loading model file: {model_file}")
            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        # Print more debug info
        print("\nDirectory contents:")
        if os.path.exists(transformer_weights_path):
            print(os.listdir(transformer_weights_path))
        return None, None

# Test the model loading
if __name__ == "__main__":
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_immigration_model()
    
    if model is not None:
        # Test on a sample text
        test_text = "The claimant from Syria applied for refugee status in Canada in 2010 citing persecution."
        print(f"\nProcessing text: {test_text}")
        
        # Process with transformer
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model(**inputs)
        print("\nModel processing completed successfully!")
