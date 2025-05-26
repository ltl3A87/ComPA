import os
import random
import math
import json

# Constants
SHAPE_TYPES = ["square", "rectangle", "right_triangle", "trapezoid"]
MAX_SHAPES = 6
GRID_SIZE_RANGE = range(3, 11)
SHAPE_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray", "cyan"]
SCALE = 8  # pixels per unit dimension

# Area calculation functions
def calculate_area(shape):
    if shape["type"] == "square":
        return shape["side"] ** 2
    elif shape["type"] == "rectangle":
        return shape["width"] * shape["height"]
    elif shape["type"] == "right_triangle":
        return 0.5 * shape["base"] * shape["height"]
    elif shape["type"] == "trapezoid":
        return 0.5 * (shape["base1"] + shape["base2"]) * shape["height"]

# Enhanced reasoning path
def generate_reasoning_path(target, nearest_shapes):
    path = [f"Target: {target['color']} {target['type']} at {target['position']} (area = {target['area']})"]
    total = target["area"]

    nearest = nearest_shapes[0]
    dist = abs(target['position'][0] - nearest['position'][0]) + abs(target['position'][1] - nearest['position'][1])
    path.append(
        f"- Nearest: {nearest['color']} {nearest['type']} at {nearest['position']} (area = {nearest['area']}), distance = {dist}"
    )
    total += nearest["area"]
    path.append(f"Sum of areas (target + nearest) = {round(total, 2)} → Rounded = {round(total)}")
    return "\n".join(path)

# Dataset generation function
def generate_sample(grid_size, sample_id):
    all_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    random.shuffle(all_positions)
    num_shapes = random.randint(2, min(MAX_SHAPES, len(all_positions)))
    selected_positions = all_positions[:num_shapes]

    shapes = []
    for i, pos in enumerate(selected_positions):
        shape_type = random.choice(SHAPE_TYPES)
        shape = {"id": f"Shape_{i}", "type": shape_type, "position": pos, "color": SHAPE_COLORS[i % len(SHAPE_COLORS)]}
        if shape_type == "square":
            shape["side"] = random.randint(1, 9)
        elif shape_type == "rectangle":
            shape["width"] = random.randint(1, 9)
            shape["height"] = random.randint(1, 9)
        elif shape_type == "right_triangle":
            shape["base"] = random.randint(1, 9)
            shape["height"] = random.randint(1, 9)
        elif shape_type == "trapezoid":
            shape["base1"] = random.randint(1, 9)
            shape["base2"] = random.randint(1, 9)
            shape["height"] = random.randint(1, 9)
        shape["area"] = round(calculate_area(shape), 2)
        shapes.append(shape)

    target = random.choice(shapes)
    k = 1

    other_shapes = [s for s in shapes if s["id"] != target["id"]]
    other_shapes.sort(key=lambda s: abs(target['position'][0] - s['position'][0]) + abs(target['position'][1] - s['position'][1]))
    nearest_shapes = other_shapes[:k]
    total_area = round(sum(s["area"] for s in nearest_shapes), 2) + target["area"]
    total_area_rounded = round(total_area)

    def describe_shape(shape):
        if shape["type"] == "square":
            return f"a {shape['color']} square at position {shape['position']} with side {shape['side']}"
        elif shape["type"] == "rectangle":
            return f"a {shape['color']} rectangle at position {shape['position']} with width {shape['width']} and height {shape['height']}"
        elif shape["type"] == "right_triangle":
            return f"a {shape['color']} right_triangle at position {shape['position']} with base {shape['base']} and height {shape['height']}"
        elif shape["type"] == "trapezoid":
            return f"a {shape['color']} trapezoid at position {shape['position']} with bases {shape['base1']}, {shape['base2']} and height {shape['height']}"

    shape_descriptions = [describe_shape(shape) for shape in shapes]

    question = (
        f"There are {len(shapes)} shapes on a {grid_size}x{grid_size} grid: " +
        ", ".join(shape_descriptions) + ". "
        f"Find the shape that is nearest to the {target['color']} {target['type']} at {target['position']} in terms of Manhattan distance. "
        f"What is the total area of this nearest shape and the target shape? Please round the result into the nearest integer."
    )

    thinking = generate_reasoning_path(target, nearest_shapes)

    metadata = {
        "sample_id": sample_id,
        "grid_size": grid_size,
        "k": k,
        "shapes": shapes,
        "target_id": target["id"],
        "selected_ids": [s["id"] for s in nearest_shapes]
    }

    return {
        "question": question,
        "answer": total_area_rounded,
        "thinking": thinking,
        "metadata": metadata
    }

# Generate multiple samples and save to one JSONL file
def generate_dataset(output_dir="shape_grid_dataset", num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")

    with open(jsonl_path, "w") as jsonl_file:
        for i in range(num_samples):
            grid_size = random.choice(GRID_SIZE_RANGE)
            sample = generate_sample(grid_size, i)
            jsonl_file.write(json.dumps(sample) + "\n")

# Example usage
if __name__ == "__main__":
    generate_dataset("./shape_spatial_reasoning_evaluation_pure_text", 500)
