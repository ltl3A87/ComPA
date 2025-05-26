from datasets import load_dataset, DatasetDict
from datasets import Image

def merge_think_answer(example):
    think_text = example["solution"].strip()
    answer_text = str(example["answer"]).strip()
    merged = f"<think>{think_text}</think>\n\n<answer>{answer_text}</answer>"
    example["solution"] = merged
    return example

# Replace with your dataset path or name
dataset = load_dataset("./mixed_mm_r1")

# Apply to each split if it's a DatasetDict (train/validation/test)
if isinstance(dataset, DatasetDict):
    dataset = dataset.map(merge_think_answer)
    dataset = dataset.remove_columns(["answer", "metadata"])
else:
    dataset = dataset.map(merge_think_answer)
    dataset = dataset.remove_columns(["answer", "metadata"])
dataset = dataset.cast_column("image", Image())

# Save or push to hub (optional)
dataset.push_to_hub("./mixed_mm_r1_merged")

# Print one sample
print(dataset["train"][0])