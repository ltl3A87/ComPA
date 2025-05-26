# import json

# input_path = "./spatial_reasoning_benchmark/spatial_dataset.jsonl"
# output_path = "./spatial_reasoning_benchmark/spatial_dataset_prefix.jsonl"
# prefix = "./spatial_reasoning_benchmark/"

# def add_prefix_to_image_path(in_path, out_path, prefix):
#     with open(in_path, "r", encoding="utf-8") as infile, open(out_path, "w", encoding="utf-8") as outfile:
#         for line in infile:
#             if not line.strip():
#                 continue
#             data = json.loads(line)
#             data["image_path"] = prefix + data["image_path"]
#             outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

# add_prefix_to_image_path(input_path, output_path, prefix)

from datasets import Dataset, Image
import re
import json

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

# Example usage
meta_json = "./spatial_reasoning_benchmark/spatial_dataset_prefix.jsonl"
dataset = load_jsonl(meta_json)

new_dataset = Dataset.from_list(dataset)

new_dataset = new_dataset.rename_column("question", "problem")
new_dataset = new_dataset.rename_column("thinking", "solution")
new_dataset = new_dataset.rename_column("image_path", "image")
new_dataset = new_dataset.cast_column("image", Image())

# (Optional) Save locally
# new_dataset.save_to_disk("pure_text_math_r1")

# (Optional) Push to HuggingFace Hub (requires authentication)
new_dataset.push_to_hub("./grid_mm_r1")