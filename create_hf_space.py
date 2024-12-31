import os
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

space_name = os.getenv("HF_SPACE_NAME")

def create_hf_space(space_name: str, private: bool = False):
    api = HfApi()
    token = os.getenv("HF_TOKEN", None)
    
    if token is None:
        raise ValueError("Hugging Face token not found. Set the HF_TOKEN environment variable.")
    
    try:
        # Check if the space already exists
        api.repo_info(repo_id=space_name, repo_type="space", token=token)
        print(f"Space '{space_name}' already exists.")
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            # Create the space if it does not exist
            api.create_repo(
                repo_id=space_name,
                repo_type="space",
                space_sdk="gradio",
                private=private,
                token=token
            )
            print(f"Space '{space_name}' created successfully.")
        else:
            raise e

if __name__ == "__main__":
    create_hf_space(space_name)
