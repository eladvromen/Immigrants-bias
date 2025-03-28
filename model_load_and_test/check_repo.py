from huggingface_hub import list_repo_files

def check_repository():
    repo_id = "clairebarale/refugee_cases_ner"
    print(f"Checking repository: {repo_id}")
    files = list_repo_files(repo_id)
    print("\nAvailable files:")
    for file in files:
        print(f"- {file}")

if __name__ == "__main__":
    check_repository() 