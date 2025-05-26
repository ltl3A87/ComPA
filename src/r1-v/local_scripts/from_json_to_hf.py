from datasets import Dataset, concatenate_datasets, Image
import json

def load_jsonl_to_dataset(path):
    """Load a jsonl file and format it to match the unified schema."""
    records = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            if item['thinking'] != "":
                records.append({
                    # "image": item["image_path"],
                    "problem": item["question"],
                    "solution": f"<think>{item['thinking']}</think>\n\n<answer>{item['answer']}</answer>"
                })
            else:
                records.append({
                    # "image": item["image_path"],
                    "problem": item["question"],
                    "solution": f"<answer>{item['answer']}</answer>"
                })
    return Dataset.from_list(records)

# Paths to your two jsonl files
path1 = "./spatial_reasoning_pure_text/spatial_dataset.jsonl"
path2 = "./shape_reasoning_pure_text/shape_dataset.jsonl"

# Load and format each dataset
ds1 = load_jsonl_to_dataset(path1)
ds2 = load_jsonl_to_dataset(path2)

# Concatenate the two datasets
# merged_ds = concatenate_datasets([ds1, ds2]).shuffle(seed=42)
merged_ds = ds1

# Cast the image column to Image()
# merged_ds = merged_ds.cast_column("image", Image())

# Optional: save to disk or push to hub
# merged_ds.save_to_disk("/your/output/path")
merged_ds.push_to_hub("./mm_r1_spatial_easy_puretext")
# merged_ds.push_to_hub("./mm_r1_spatial_easy_modify")

# Preview
print(merged_ds[0])