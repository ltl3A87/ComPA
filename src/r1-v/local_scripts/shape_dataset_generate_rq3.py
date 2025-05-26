import random
import math
import json
from PIL import Image, ImageDraw, ImageFont
import os

# Output
output_dir = "./shape_reasoning_evaluation_rq3"
image_output_dir = os.path.join(output_dir, "images")
os.makedirs(image_output_dir, exist_ok=True)

font = ImageFont.load_default()
# shape_types = ["square", "rectangle", "circle", "equilateral_triangle", "right_triangle", "pentagon", "hexagon", "trapezoid"]

# shape_types = ["square", "rectangle", "equilateral_triangle", "right_triangle", "trapezoid"]
shape_types = ["square", "rectangle", "right_triangle", "trapezoid"]

def random_color():
    return tuple(random.randint(50, 220) for _ in range(3))

def draw_text_safe(draw, text, x, y, image_size):
    """Draw text above or below the shape to avoid overlap, using text bounding box"""
    bbox = font.getbbox(text)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Try placing the label above the shape
    if y - text_height > 0:
        draw.text((x, y - text_height - 4), text, fill="black", font=font)
    else:
        # Otherwise place below
        draw.text((x, min(y + 4, image_size - text_height)), text, fill="black", font=font)

def compute_properties(shape, props):
    """Returns (LaTeX thinking step, area value)"""
    if shape == "square":
        s = props["side"]
        area = s * s
        return f"$\\text{{Square: }} s={s} \\Rightarrow A=s^2={s}^2={area}$", area
    elif shape == "rectangle":
        w, h = props["width"], props["height"]
        area = w * h
        return f"$\\text{{Rectangle: }} w={w}, h={h} \\Rightarrow A=w \\times h = {w} \\times {h} = {area}$", area
    elif shape == "circle":
        r = props["radius"]
        area = round(math.pi * r * r, 2)
        return f"$\\text{{Circle: }} r={r} \\Rightarrow A=\\pi \\times r^2 = \\pi \\times {r}^2 = {area}$", area
    elif shape == "equilateral_triangle":
        s = props["side"]
        area = round((math.sqrt(3)/4) * s * s, 2)
        return f"$\\text{{Equilateral triangle: }} s={s} \\Rightarrow A=\\frac{{\\sqrt{{3}}}}{4} \\times s^2 = \\frac{{\\sqrt{{3}}}}{4} \\times {s}^2 = {area}$", area
    elif shape == "right_triangle":
        a, b = props["a"], props["b"]
        area = 0.5 * a * b
        return f"$\\text{{Right triangle: }} a={a}, b={b} \\Rightarrow A=\\frac{{1}}{{2}} \\times a \\times b = \\frac{{1}}{{2}} \\times {a} \\times {b} = {area}$", area
    elif shape == "pentagon":
        s = props["side"]
        area = round((5 * s * s) / (4 * math.tan(math.pi / 5)), 2)
        return f"$\\text{{Pentagon: }} s={s} \\Rightarrow A=\\frac{{5s^2}}{{4\\tan(\\pi/5)}} = \\frac{{5\\times{s}^2}}{{4\\tan(\\pi/5)}} = {area}$", area
    elif shape == "hexagon":
        s = props["side"]
        area = round((3 * math.sqrt(3) / 2) * s * s, 2)
        return f"$\\text{{Hexagon: }} s={s} \\Rightarrow A=\\frac{{3\\sqrt{{3}}}}{2} \\times s^2 = \\frac{{3\\sqrt{{3}}}}{2} \\times {s}^2 = {area}$", area
    elif shape == "trapezoid":
        a, b, h = props["a"], props["b"], props["height"]
        area = 0.5 * (a + b) * h
        return f"$\\text{{Trapezoid: }} a={a}, b={b}, h={h} \\Rightarrow A=\\frac{{a+b}}{2} \\times h = \\frac{{{a}+{b}}}{2} \\times {h} = {area}$", area

def generate_one_example(idx, image_size=512):
    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)

    n_shapes = random.randint(2, 6)
    thinking_steps = []
    max_area = 0
    label_boxes = []
    labels_to_draw = []

    for _ in range(n_shapes):
        shape = random.choice(shape_types)
        x = random.randint(50, image_size - 150)
        y = random.randint(50, image_size - 150)
        color = random_color()
        props = {}
        label_text = ""

        if shape == "square":
            s = props["side"] = random.randint(1, 9)
            draw.rectangle([x, y, x + s * 10, y + s * 10], fill=color)
            label_text = f"s={s}"

        elif shape == "rectangle":
            w = props["width"] = random.randint(1, 9)
            h = props["height"] = random.randint(1, 9)
            draw.rectangle([x, y, x + w * 10, y + h * 10], fill=color)
            label_text = f"{w} x {h}"

        elif shape == "right_triangle":
            a = props["a"] = random.randint(1, 9)
            b = props["b"] = random.randint(1, 9)
            draw.polygon([(x, y), (x, y + b * 10), (x + a * 10, y + b * 10)], fill=color)
            label_text = f"{a} x {b}"

        elif shape == "trapezoid":
            a = props["a"] = random.randint(1, 9)
            b = props["b"] = random.randint(1, 9)
            h = props["height"] = random.randint(1, 9)
            points = [
                (x, y),
                (x + a * 10, y),
                (x + b * 10 + 10, y + h * 10),
                (x - 10, y + h * 10)
            ]
            draw.polygon(points, fill=color)
            label_text = f"{a}, {b}, h={h}"

        step_text, area = compute_properties(shape, props)
        max_area = max(max_area, area)
        thinking_steps.append(step_text)
        labels_to_draw.append((label_text, x, y))

    for label_text, x, y in labels_to_draw:
        bbox = font.getbbox(label_text)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        label_x = x
        label_y = y - text_h - 4

        while any(abs(label_x - bx) < text_w and abs(label_y - by) < text_h for bx, by, bw, bh in label_boxes):
            label_y += text_h + 2
            if label_y + text_h > image_size:
                label_y = max(0, y + 4)

        draw.text((label_x, label_y), label_text, fill="black", font=font)
        label_boxes.append((label_x, label_y, text_w, text_h))

    question = "What is the largest area among all shapes shown in the image? Please round the result into the nearest integer."
    thinking = "\n".join(thinking_steps)
    answer = round(max_area)

    image_path = os.path.join(image_output_dir, f"shape_{idx:05d}.png")
    img.save(image_path)

    return {
        "image_path": image_path,
        "question": question,
        "thinking": thinking,
        "answer": f"{answer}"
    }

# Generate dataset
with open(os.path.join(output_dir, "shape_dataset.jsonl"), "w") as f:
    for i in range(500):
        item = generate_one_example(i)
        f.write(json.dumps(item) + "\n")

print("✅ Modified shape benchmark dataset generated!")