import os
import json
import re
from datasets import load_dataset

# Configuration
# dataset = load_dataset("./mm_math_r1", split="train").select(range(500))

import json

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

# Example usage
meta_json = "./shape_reasoning_evaluation_ground/shape_dataset.jsonl"
dataset = load_jsonl(meta_json)

# image_dir = "./spatial_reasoning_evaluation_new/images"
output_jsonl = "./src/eval/prompts/shape-area-ground.jsonl"

def extract_answer_from_solution(solution_text):
    match = re.search(r"<answer>(.*?)</answer>", solution_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for idx, example in enumerate(dataset):
        if idx == 0:
            print(example)
        # image_path = os.path.join(image_dir, f"problem_{idx:05d}.png")
        # image = example["image"]
        # image.save(image_path)
        image_path = example["image_path"]
        question = example["question"] # + "\nYou need to find the nearest k shapes according to the question to the target shape in terms of Manhattan distance and then sum up their total area."
        # question = example["problem"]
        # question = "Answer the question shown in the image."
        # solution = example.get("solution", "")
        # ground_truth = extract_answer_from_solution(solution)
        ground_truth = example.get("answer", "")

        entry = {
            "image_path": image_path,
            "question": question,
            "ground_truth": ground_truth
        }
        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Done! Saved to: {output_jsonl}")