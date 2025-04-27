import spacy
import spacy_transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer
import os
import json

# Set environment variables
os.environ["HF_HOME"] = "/data/resource/huggingface"

# Path to the scratch_legalbert model directory
model_path = "/data/shil6369/legal_bert_project/models/asylex_model/scratch_legalbert/model-last"

def load_scratch_legalbert_model():
    try:
        print("\nChecking spaCy and transformer components...")
        # Get available factories in a compatible way
        print("Available spaCy factories:", list(spacy.registry.factories.get_all().keys()))
        
        print("\nAttempting to load spaCy model...")
        print(f"Loading from path: {model_path}")
        
        # Try to read config first
        config_path = os.path.join(model_path, "config.cfg")
        if os.path.exists(config_path):
            print(f"Found config at: {config_path}")
            with open(config_path, 'r') as f:
                print("Config contents:", f.read())
        
        nlp = spacy.load(model_path)
        print("SpaCy model loaded successfully!")
        
        return nlp
    except Exception as e:
        print(f"\nDetailed error: {str(e)}")
        print("\nDirectory structure:")
        if os.path.exists(model_path):
            print("model-last contents:", os.listdir(model_path))
            # Print contents of transformer directory
            transformer_path = os.path.join(model_path, "transformer")
            if os.path.exists(transformer_path):
                print("transformer contents:", os.listdir(transformer_path))
        return None

if __name__ == "__main__":
    print(f"Loading model from {model_path}...")
    nlp = load_scratch_legalbert_model()
    
    if nlp is not None:
        # Test spaCy model
        test_text = "The claimant from Syria applied for refugee status in Canada in 2010 citing persecution."
        print(f"\nProcessing text with spaCy: {test_text}")
        doc = nlp(test_text)
        print("\nNamed Entities found:")
        for ent in doc.ents:
            print(f"- {ent.text} ({ent.label_})")
