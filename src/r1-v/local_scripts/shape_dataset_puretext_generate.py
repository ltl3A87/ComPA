import random
import math
import json
from PIL import ImageDraw, ImageFont
import os

# Output
output_dir = "./shape_reasoning_pure_text_evaluation"
os.makedirs(output_dir, exist_ok=True)

font = ImageFont.load_default()
shape_types = ["square", "rectangle", "right_triangle", "trapezoid"]


def compute_properties(shape, props):
    if shape == "square":
        s = props["side"]
        area = s * s
        return f"Square with side {s}: Area = {s}^2 = {area}", area
    elif shape == "rectangle":
        w, h = props["width"], props["height"]
        area = w * h
        return f"Rectangle with width {w} and height {h}: Area = {w} x {h} = {area}", area
    elif shape == "right_triangle":
        a, b = props["a"], props["b"]
        area = 0.5 * a * b
        return f"Right triangle with base {a} and height {b}: Area = 0.5 x {a} x {b} = {area}", area
    elif shape == "trapezoid":
        a, b, h = props["a"], props["b"], props["height"]
        area = 0.5 * (a + b) * h
        return f"Trapezoid with bases {a}, {b} and height {h}: Area = 0.5 x ({a} + {b}) x {h} = {area}", area


def generate_one_example(idx):
    n_shapes = random.randint(2, 6)
    thinking_steps = []
    shape_descriptions = []
    total = 0

    for _ in range(n_shapes):
        shape = random.choice(shape_types)
        props = {}
        label_text = ""

        if shape == "square":
            props["side"] = random.randint(1, 9)
        elif shape == "rectangle":
            props["width"] = random.randint(1, 9)
            props["height"] = random.randint(1, 9)
        elif shape == "right_triangle":
            props["a"] = random.randint(1, 9)
            props["b"] = random.randint(1, 9)
        elif shape == "trapezoid":
            props["a"] = random.randint(1, 9)
            props["b"] = random.randint(1, 9)
            props["height"] = random.randint(1, 9)

        step_text, value = compute_properties(shape, props)
        total += value
        thinking_steps.append(step_text)

        if shape == "square":
            shape_descriptions.append(f"a {shape} with side {props['side']}")
        elif shape == "rectangle":
            shape_descriptions.append(f"a {shape} with width {props['width']} and height {props['height']}")
        elif shape == "right_triangle":
            shape_descriptions.append(f"a {shape} with base {props['a']} and height {props['b']}")
        elif shape == "trapezoid":
            shape_descriptions.append(f"a {shape} with bases {props['a']} and {props['b']} and height {props['height']}")

    question = (
        "Given the following shapes: " + ", ".join(shape_descriptions) + ". "
        "What is the total area of all shapes? Please round the result to the nearest integer."
    )
    thinking = "\n".join(thinking_steps)
    answer = round(total)

    return {
        "question": question,
        "thinking": thinking,
        "answer": f"{answer}"
    }


# Generate dataset
with open(os.path.join(output_dir, "shape_dataset.jsonl"), "w") as f:
    for i in range(500):
        item = generate_one_example(i)
        f.write(json.dumps(item) + "\n")

print("✅ Text-based shape benchmark dataset generated!")
