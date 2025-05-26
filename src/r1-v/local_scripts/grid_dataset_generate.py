import os
import random
import math
import json
from PIL import Image, ImageDraw, ImageFont

# Output directories
output_dir = "./spatial_reasoning_evaluation_harder"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

# Constants
# shape_types = ["square", "rectangle", "circle", "equilateral_triangle", "right_triangle", "pentagon", "hexagon", "trapezoid"]
# shape_types = ["square", "rectangle", "equilateral_triangle", "right_triangle", "trapezoid"]
shape_types = ["square", "rectangle", "right_triangle", "trapezoid"]
color_names = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "cyan", "gray"]
color_values = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
    "cyan": (0, 255, 255),
    "gray": (128, 128, 128),
}
font = ImageFont.load_default()

def draw_shape(draw, shape, bbox, color):
    x, y, cell_w, cell_h = bbox
    cx = x + cell_w // 2
    cy = y + cell_h // 2
    size = int(min(cell_w, cell_h) * 0.6)  # keep size proportionally smaller

    # Make sure the shape fits within the cell (adjust size for polygons)
    max_size = min(cell_w, cell_h) * 0.6  # Max size to ensure no overflow

    if shape == "square":
        offset = size // 2
        draw.rectangle([cx - offset, cy - offset, cx + offset, cy + offset], fill=color)
    elif shape == "rectangle":
        draw.rectangle([cx - size//2, cy - size//3, cx + size//2, cy + size//3], fill=color)
    elif shape == "circle":
        draw.ellipse([cx - size//2, cy - size//2, cx + size//2, cy + size//2], fill=color)
    elif shape == "equilateral_triangle":
        h = int((math.sqrt(3)/2) * size)
        points = [(cx, cy - h // 2), (cx - size // 2, cy + h // 2), (cx + size // 2, cy + h // 2)]
        draw.polygon(points, fill=color)
    elif shape == "right_triangle":
        points = [(cx - size//2, cy + size//2), (cx - size//2, cy - size//2), (cx + size//2, cy + size//2)]
        draw.polygon(points, fill=color)
    elif shape == "pentagon":
        # Adjust size to fit the grid cell
        s = size * 0.8  # Ensure pentagon fits inside the cell
        points = [(cx + s * math.cos(2 * math.pi * i / 5),
                   cy + s * math.sin(2 * math.pi * i / 5)) for i in range(5)]
        draw.polygon(points, fill=color)
    elif shape == "hexagon":
        # Adjust size to fit the grid cell
        s = size * 0.8  # Ensure hexagon fits inside the cell
        points = [(cx + s * math.cos(2 * math.pi * i / 6),
                   cy + s * math.sin(2 * math.pi * i / 6)) for i in range(6)]
        draw.polygon(points, fill=color)
    elif shape == "trapezoid":
        a = size
        b = size * 0.6
        h = size * 0.6
        points = [(cx - a//2, cy - h//2), (cx + a//2, cy - h//2),
                  (cx + b//2, cy + h//2), (cx - b//2, cy + h//2)]
        draw.polygon(points, fill=color)

def generate_example(index):
    grid_size = random.randint(10, 16)
    shape_count = random.randint(2, min(6, grid_size * grid_size))
    img_size = 512
    cell_size = img_size // grid_size

    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    # Draw grid lines
    for i in range(grid_size + 1):
        y = i * cell_size
        draw.line([(0, y), (img_size, y)], fill="black", width=1)
    for j in range(grid_size + 1):
        x = j * cell_size
        draw.line([(x, 0), (x, img_size)], fill="black", width=1)

    positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(positions)
    selected_positions = positions[:shape_count]

    available_colors = random.sample(color_names, shape_count)
    used_shapes = random.choices(shape_types, k=shape_count)

    shape_data = []

    for idx, (shape, color_name, (i, j)) in enumerate(zip(used_shapes, available_colors, selected_positions)):
        x = j * cell_size
        y = i * cell_size
        draw_shape(draw, shape, (x, y, cell_size, cell_size), color_values[color_name])
        shape_data.append({
            "index": idx,
            "shape": shape,
            "color": color_name,
            "grid_position": (i, j),
        })

    # Choose a target shape randomly
    target_idx = random.randint(0, shape_count - 1)
    target_pos = shape_data[target_idx]["grid_position"]

    # Initialize thinking path
    thinking_path = [f"Target shape: {shape_data[target_idx]['color']} {shape_data[target_idx]['shape']} at {target_pos}"]

    # Find the closest shape to the target (excluding itself)
    def manhattan(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    closest_shape = None
    min_distance = float("inf")

    for data in shape_data:
        if data["index"] == target_idx:
            continue
        dist = manhattan(data["grid_position"], target_pos)
        thinking_path.append(f"Distance from target shape to {data['color']} {data['shape']} at {data['grid_position']}: {dist}")
        if dist < min_distance:
            min_distance = dist
            closest_shape = data

    # Add the final step to the thinking path
    final_step = f"Closest shape is the {closest_shape['color']} {closest_shape['shape']} at {closest_shape['grid_position']} with distance {min_distance}"
    thinking_path.append(final_step)

    # Build question
    target_shape = shape_data[target_idx]
    question = (
        f"There are {shape_count} shapes on a {grid_size}x{grid_size} grid. "
        f"Which shape is closest to the {target_shape['color']} {target_shape['shape']} in terms of Manhattan distance? The final answer is the grid position of the closest shape like (row_index, column_index)."
    )

    return {
        "image_path": f"{output_dir}/images/example_{index:05d}.png",
        "question": question,
        "answer": str(closest_shape["grid_position"]),  # now a tuple like (2, 4)
        "thinking": "\n".join(thinking_path),  # thinking path included
        "metadata": {
            "target_index": target_idx,
            "target": target_shape,
            "closest_shape": closest_shape,
            "shapes": shape_data,
            "grid_size": grid_size,
        }
    }, img

# Generate dataset
jsonl_path = os.path.join(output_dir, "spatial_dataset.jsonl")
with open(jsonl_path, "w") as f:
    for i in range(500):  # change to your desired count
        data, image = generate_example(i)
        image.save(os.path.join(output_dir, data["image_path"]))
        f.write(json.dumps(data) + "\n")

print("✅ Spatial reasoning benchmark created!")