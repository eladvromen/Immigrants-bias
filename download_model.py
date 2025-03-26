from huggingface_hub import snapshot_download
import os

print("Downloading model files...")
output_dir = "/data/shil6369/legal_bert_project/models/asylex_model"
os.makedirs(output_dir, exist_ok=True)

# Download the specific subdirectory
path = snapshot_download(
    repo_id="clairebarale/refugee_cases_ner",
    allow_patterns=["pretrained_legalbert/model-last/**", "pretrained_legalbert/config.cfg", "pretrained_legalbert/meta.json"],
    local_dir=output_dir,
    local_dir_use_symlinks=False
)

print(f"Files downloaded to: {path}")
print("Directory structure:")
for root, dirs, files in os.walk(output_dir):
    rel_path = os.path.relpath(root, output_dir)
    if rel_path == '.':
        continue
    level = rel_path.count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = ' ' * 4 * (level + 1)
    for f in files:
        print(f"{sub_indent}{f}")
