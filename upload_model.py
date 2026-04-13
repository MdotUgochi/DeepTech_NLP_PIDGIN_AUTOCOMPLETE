from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="pidgin-autocomplete-model",
    repo_id="Ugochief/GPT2_pidgin_autocomplete",
    repo_type="model"
)