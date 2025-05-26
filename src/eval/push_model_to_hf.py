from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor  # replace with actual import
import torch

def init_model(model_path, gpu_id, push=False, hf_repo_name=None):
    """Init a model on a specific GPU, optionally push to Hugging Face Hub."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu_id}",
    )

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    if push and hf_repo_name:
        print(f"Pushing model and processor to Hugging Face: {hf_repo_name}")
        model.push_to_hub(hf_repo_name)
        processor.push_to_hub(hf_repo_name)

    return model, processor


