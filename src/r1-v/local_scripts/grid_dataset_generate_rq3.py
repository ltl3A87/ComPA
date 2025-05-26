import os
import random
import math
import json
from PIL import Image, ImageDraw, ImageFont

# Output directories
output_dir = "./spatial_reasoning_evaluation_rq3"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

shape_types = ["square", "rectangle", "right_triangle", "trapezoid"]
color_names = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "cyan", "gray"]
color_values = {"red": (255,0,0), "blue": (0,0,255), "green": (0,128,0), "yellow": (255,255,0), "purple": (128,0,128), "orange": (255,165,0), "pink": (255,105,180), "brown": (139,69,19), "cyan": (0,255,255), "gray": (128,128,128)}
font = ImageFont.load_default()

def draw_shape(draw, shape, bbox, color):
    x, y, cell_w, cell_h = bbox
    cx, cy = x + cell_w // 2, y + cell_h // 2
    size = int(min(cell_w, cell_h) * 0.6)

    if shape == "square":
        offset = size // 2
        draw.rectangle([cx - offset, cy - offset, cx + offset, cy + offset], fill=color)
    elif shape == "rectangle":
        draw.rectangle([cx - size//2, cy - size//3, cx + size//2, cy + size//3], fill=color)
    elif shape == "right_triangle":
        points = [(cx - size//2, cy + size//2), (cx - size//2, cy - size//2), (cx + size//2, cy + size//2)]
        draw.polygon(points, fill=color)
    elif shape == "trapezoid":
        a, b, h = size, size * 0.6, size * 0.6
        points = [(cx - a//2, cy - h//2), (cx + a//2, cy - h//2), (cx + b//2, cy + h//2), (cx - b//2, cy + h//2)]
        draw.polygon(points, fill=color)

def generate_example(index):
    grid_size = random.randint(3, 9)
    shape_count = random.randint(2, min(6, grid_size**2))
    img_size, cell_size = 512, 512 // grid_size

    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    for i in range(grid_size + 1):
        draw.line([(0, i*cell_size), (img_size, i*cell_size)], fill="black")
        draw.line([(i*cell_size, 0), (i*cell_size, img_size)], fill="black")

    positions = random.sample([(i,j) for i in range(grid_size) for j in range(grid_size)], shape_count)
    colors = random.sample(color_names, shape_count)
    shapes = random.choices(shape_types, k=shape_count)

    shape_data = []
    for idx, (shape, color, (i,j)) in enumerate(zip(shapes, colors, positions)):
        draw_shape(draw, shape, (j*cell_size, i*cell_size, cell_size, cell_size), color_values[color])
        shape_data.append({"index": idx, "shape": shape, "color": color, "grid_position": (i,j)})

    target_idx = random.randint(0, shape_count - 1)
    target_pos = shape_data[target_idx]["grid_position"]

    thinking_path = [f"Target shape: {shape_data[target_idx]['color']} {shape_data[target_idx]['shape']} at {target_pos}"]

    def manhattan(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    farthest_shape, max_dist = None, -1

    for data in shape_data:
        if data["index"] == target_idx:
            continue
        dist = manhattan(data["grid_position"], target_pos)
        thinking_path.append(f"Distance from target shape to {data['color']} {data['shape']} at {data['grid_position']}: {dist}")
        if dist > max_dist:
            max_dist = dist
            farthest_shape = data

    thinking_path.append(f"Farthest shape is the {farthest_shape['color']} {farthest_shape['shape']} at {farthest_shape['grid_position']} with distance {max_dist}")

    question = (f"There are {shape_count} shapes on a {grid_size}x{grid_size} grid. "
                f"Which shape is farthest from the {shape_data[target_idx]['color']} {shape_data[target_idx]['shape']} in terms of Manhattan distance? "
                "The final answer is the grid position of the farthest shape like (row_index, column_index).")

    return {"image_path": f"{output_dir}/images/example_{index:05d}.png",
            "question": question,
            "answer": str(farthest_shape["grid_position"]),
            "thinking": "\n".join(thinking_path),
            "metadata": {"target_index": target_idx, "target": shape_data[target_idx], "farthest_shape": farthest_shape, "shapes": shape_data, "grid_size": grid_size}}, img

with open(os.path.join(output_dir, "spatial_farthest_dataset.jsonl"), "w") as f:
    for i in range(500):
        data, img = generate_example(i)
        img.save(data["image_path"])
        f.write(json.dumps(data) + "\n")

print("✅ Farthest spatial reasoning dataset created!")