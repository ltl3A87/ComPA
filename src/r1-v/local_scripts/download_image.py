import os
from datasets import load_dataset
from PIL import Image

# Load the dataset
dataset = load_dataset("./leonardPKU/GEOQA_R1V_Train_8K")

# Specify the output directory
output_dir = "./Geo170K/images/test/"
os.makedirs(output_dir, exist_ok=True)

# Iterate through the dataset and save images
for idx, example in enumerate(dataset["train"]):
    image: Image.Image = example["image"]
    image_path = os.path.join(output_dir, f"{idx}.png")
    image.save(image_path)
    if idx == 500:
        break

print(f"✅ Saved {len(dataset['train'])} images to {output_dir}")