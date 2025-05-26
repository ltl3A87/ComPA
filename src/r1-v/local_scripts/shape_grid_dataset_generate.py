import os
import random
import math
import json
from PIL import Image, ImageDraw, ImageFont

# Constants
SHAPE_TYPES = ["square", "rectangle", "right_triangle", "trapezoid"]
MAX_SHAPES = 6
GRID_SIZE_RANGE = range(3, 11)
SHAPE_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray", "cyan"]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
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

# Draw grid lines
def draw_grid(draw, grid_size, cell_size):
    for i in range(grid_size + 1):
        draw.line([(i * cell_size, 0), (i * cell_size, grid_size * cell_size)], fill="black", width=2)
        draw.line([(0, i * cell_size), (grid_size * cell_size, i * cell_size)], fill="black", width=2)

# Centered text drawing
def draw_text_centered(draw, text, box, font):
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box[0][0], box[0][1], box[1][0], box[1][1]
    center_x = x0 + (x1 - x0 - text_width) / 2
    center_y = y0 + (y1 - y0 - text_height) / 2
    draw.text((center_x, center_y), text, fill="black", font=font)

# Shape drawing using true dimensions, centered in the grid cell
def draw_shape(draw, shape, cell_size, font):
    x, y = shape["position"]
    grid_cx, grid_cy = x * cell_size, y * cell_size
    center_x, center_y = grid_cx + cell_size / 2, grid_cy + cell_size / 2

    if shape["type"] == "square":
        size = shape["side"] * SCALE
        top_left = (center_x - size / 2, center_y - size / 2)
        bottom_right = (center_x + size / 2, center_y + size / 2)
        shape_box = [top_left, bottom_right]
        draw.rectangle(shape_box, fill=shape["color"])
        label_text = f"s={shape['side']}"

    elif shape["type"] == "rectangle":
        w = shape["width"] * SCALE
        h = shape["height"] * SCALE
        top_left = (center_x - w / 2, center_y - h / 2)
        bottom_right = (center_x + w / 2, center_y + h / 2)
        shape_box = [top_left, bottom_right]
        draw.rectangle(shape_box, fill=shape["color"])
        label_text = f"{shape['width']}x{shape['height']}"

    elif shape["type"] == "right_triangle":
        b = shape["base"] * SCALE
        h = shape["height"] * SCALE
        base_left = center_x - b / 2
        base_right = center_x + b / 2
        top_y = center_y - h / 2
        bottom_y = center_y + h / 2
        shape_box = [(base_left, top_y), (base_right, bottom_y)]
        draw.polygon([(base_left, bottom_y), (base_right, bottom_y), (base_left, top_y)], fill=shape["color"])
        label_text = f"{shape['base']}x{shape['height']}"

    elif shape["type"] == "trapezoid":
        b1 = shape["base1"] * SCALE
        b2 = shape["base2"] * SCALE
        h = shape["height"] * SCALE
        max_base = max(b1, b2)
        cx = center_x - max_base / 2
        cy = center_y - h / 2
        top_left = (cx + (max_base - b1) / 2, cy)
        top_right = (top_left[0] + b1, cy)
        bottom_left = (cx + (max_base - b2) / 2, cy + h)
        bottom_right = (bottom_left[0] + b2, cy + h)
        draw.polygon([top_left, top_right, bottom_right, bottom_left], fill=shape["color"])
        shape_box = [top_left, bottom_right]
        label_text = f"{shape['base1']},{shape['base2']},h={shape['height']}"

    draw_text_centered(draw, label_text, shape_box, font)

# Enhanced reasoning path

# def generate_reasoning_path(target, nearest_shapes):
#     path = [f"Target: {target['color']} {target['type']} at {target['position']}"]
#     total = 0
#     for shape in nearest_shapes:
#         dist = abs(target['position'][0] - shape['position'][0]) + abs(target['position'][1] - shape['position'][1])
#         path.append(
#             f"- {shape['color']} {shape['type']} at {shape['position']}: distance = |{target['position'][0]} - {shape['position'][0]}| + |{target['position'][1]} - {shape['position'][1]}| = {dist}, area = {shape['area']}"
#         )
#         total += shape['area']
#     path.append(f"Sum of areas = {round(total, 2)} → Rounded = {round(total)}")
#     return "\n".join(path)
def generate_reasoning_path(target, nearest_shapes):
    path = [f"Target: {target['color']} {target['type']} at {target['position']}"]
    total = target["area"]
    path.append(f"- {target['color']} {target['type']} at {target['position']}: area = {target['area']}")

    nearest = nearest_shapes[0]
    dist = abs(target['position'][0] - nearest['position'][0]) + abs(target['position'][1] - nearest['position'][1])
    path.append(
        f"- {nearest['color']} {nearest['type']} at {nearest['position']}: distance = "
        f"|{target['position'][0]} - {nearest['position'][0]}| + "
        f"|{target['position'][1]} - {nearest['position'][1]}| = {dist}, area = {nearest['area']}"
    )
    total += nearest["area"]
    path.append(f"Sum of areas (target + nearest) = {round(total, 2)} → Rounded = {round(total)}")
    return "\n".join(path)

# Dataset generation function
def generate_sample(draw, grid_size, cell_size, sample_id, images_dir, font):
    draw_grid(draw, grid_size, cell_size)

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
        draw_shape(draw, shape, cell_size, font)
        shapes.append(shape)

    target = random.choice(shapes)
    # k = random.randint(1, min(len(shapes) - 1, 5))
    k = 1

    other_shapes = [s for s in shapes if s["id"] != target["id"]]
    other_shapes.sort(key=lambda s: abs(target['position'][0] - s['position'][0]) + abs(target['position'][1] - s['position'][1]))
    nearest_shapes = other_shapes[:k]
    total_area = round(sum(s["area"] for s in nearest_shapes), 2) + target["area"]
    total_area_rounded = round(total_area)

    # question = f"What is the total area of the {k} nearest shapes to the {target['color']} {target['type']}? (Round the result to the nearest integer)"
    # question = f"What is the total area of the nearest shapes to the {target['color']} {target['type']}? (Round the result to the nearest integer)"
    question = (
        f"There are {len(shapes)} shapes on a {grid_size}x{grid_size} grid. "
        f"Find the shape that is nearest to the {target['color']} {target['type']} in terms of Manhattan distance, what is the total area of the nearest shape to the target shape and the target shape? Please round the result into the nearest integer."
    )
    thinking = generate_reasoning_path(target, nearest_shapes)

    image_filename = f"sample_{sample_id}.png"
    image_path = os.path.abspath(os.path.join(images_dir, image_filename))

    metadata = {
        "sample_id": sample_id,
        "grid_size": grid_size,
        "k": k,
        "shapes": shapes,
        "target_id": target["id"],
        "selected_ids": [s["id"] for s in nearest_shapes]
    }

    return {
        "image_path": image_path,
        "question": question,
        "answer": total_area_rounded,
        "thinking": thinking,
        "metadata": metadata
    }, image_filename

# Generate multiple samples and save to one JSONL file
def generate_dataset(output_dir="shape_grid_dataset", num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    cell_size = 100

    try:
        font = ImageFont.truetype(FONT_PATH, size=12)
    except IOError:
        font = ImageFont.load_default()

    with open(jsonl_path, "w") as jsonl_file:
        for i in range(num_samples):
            grid_size = random.choice(GRID_SIZE_RANGE)
            image_size = (grid_size * cell_size, grid_size * cell_size)
            image = Image.new("RGB", image_size, "white")
            draw = ImageDraw.Draw(image)
            sample, image_filename = generate_sample(draw, grid_size, cell_size, i, images_dir, font)
            image.save(os.path.join(images_dir, image_filename))
            jsonl_file.write(json.dumps(sample) + "\n")

# Example usage
if __name__ == "__main__":
    generate_dataset("./shape_spatial_reasoning_evaluation_simple", 500)
