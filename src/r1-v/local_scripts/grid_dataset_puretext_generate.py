import os
import random
import math
import json
from PIL import Image, ImageDraw, ImageFont

# Output directories
output_dir = "./spatial_reasoning_pure_text_evaluation"
os.makedirs(output_dir, exist_ok=True)

# Constants
shape_types = ["square", "rectangle", "right_triangle", "trapezoid"]
color_names = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "cyan", "gray"]


def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def generate_text_example(index):
    grid_size = random.randint(3, 9)
    shape_count = random.randint(2, min(6, grid_size * grid_size))

    positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(positions)
    selected_positions = positions[:shape_count]

    available_colors = random.sample(color_names, shape_count)
    used_shapes = random.choices(shape_types, k=shape_count)

    shape_data = []
    for idx, (shape, color_name, (i, j)) in enumerate(zip(used_shapes, available_colors, selected_positions)):
        shape_data.append({
            "index": idx,
            "shape": shape,
            "color": color_name,
            "grid_position": (i, j),
        })

    # Choose a target shape randomly
    target_idx = random.randint(0, shape_count - 1)
    target_shape = shape_data[target_idx]
    target_pos = target_shape["grid_position"]

    # Build the text question
    shape_descriptions = [
        f"a {entry['color']} {entry['shape']} at position {entry['grid_position']}"
        for entry in shape_data
    ]

    question = (
        f"There are {shape_count} shapes on a {grid_size}x{grid_size} grid: " + 
        ", ".join(shape_descriptions) + ". "
        f"Which shape is closest to the {target_shape['color']} {target_shape['shape']} at position {target_shape['grid_position']} in terms of Manhattan distance? "
        f"The final answer should be the grid position of the closest shape like (row_index, column_index)."
    )

    # Find the closest shape to the target (excluding itself)
    thinking_path = [
        f"Target shape: {target_shape['color']} {target_shape['shape']} at {target_pos}"
    ]

    closest_shape = None
    min_distance = float("inf")
    for data in shape_data:
        if data["index"] == target_idx:
            continue
        dist = manhattan(data["grid_position"], target_pos)
        thinking_path.append(f"Distance from target to {data['color']} {data['shape']} at {data['grid_position']}: {dist}")
        if dist < min_distance:
            min_distance = dist
            closest_shape = data

    thinking_path.append(
        f"Closest shape is the {closest_shape['color']} {closest_shape['shape']} at {closest_shape['grid_position']} with distance {min_distance}"
    )

    return {
        "question": question,
        "answer": str(closest_shape["grid_position"]),
        "thinking": "\n".join(thinking_path),
        "metadata": {
            "target_index": target_idx,
            "target": target_shape,
            "closest_shape": closest_shape,
            "shapes": shape_data,
            "grid_size": grid_size,
        }
    }


# Generate dataset
jsonl_path = os.path.join(output_dir, "spatial_dataset.jsonl")
with open(jsonl_path, "w") as f:
    for i in range(500):
        data = generate_text_example(i)
        f.write(json.dumps(data) + "\n")

print("✅ Text-based spatial reasoning benchmark created!")
