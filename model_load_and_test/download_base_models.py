import os
from transformers import AutoModel, AutoTokenizer

# Ensure HF cache location is set
os.environ["HF_HOME"] = "/data/resource/huggingface"

# Download base BERT
print("Downloading base BERT...")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
print("Base BERT downloaded successfully")

# Download LegalBERT
print("Downloading LegalBERT...")
legal_bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
print("LegalBERT downloaded successfully")

print("Both models have been cached to:", os.environ["HF_HOME"])
