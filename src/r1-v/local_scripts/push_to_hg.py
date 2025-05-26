from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch

model_path = "./outputs/qwenvl_25_7B_mix_R1V_Train_8K_wokl"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:0",
    )

    # default processer
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

from huggingface_hub import HfApi, HfFolder, Repository

# Replace with your desired repo name and username/org
repo_name = "qwenvl_25_7B_mix_R1V_Train_8K_wokl"
hf_username = "."  # e.g., "myname" or "myorg"
full_repo_name = f"{hf_username}/{repo_name}"

# Step 1: Create the repo on the hub
from huggingface_hub import create_repo
create_repo(full_repo_name, private=False, exist_ok=True)

# Step 2: Upload model and processor
from transformers import AutoModelForCausalLM, AutoProcessor

model.push_to_hub(full_repo_name)
processor.push_to_hub(full_repo_name)
