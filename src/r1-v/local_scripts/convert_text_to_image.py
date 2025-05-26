import matplotlib.pyplot as plt
from datasets import load_dataset, Image
import os
from PIL import Image, ImageDraw, ImageFont


# Load your dataset
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")

dataset = dataset.select(range(500))

# Output directory
output_dir = ".../ComPA/math_problems_latex_rendered_open-r1-math-500"
os.makedirs(output_dir, exist_ok=True)

# Setup matplotlib for mathtext rendering
plt.rcParams.update({
    "font.size": 20,
    "figure.figsize": (6, 5),  # width, height in inches (auto-scaled later)
    "text.usetex": True,
    "font.family": "serif",
})

import re

# def escape_latex_text(text):
#     parts = re.split(r"(\$.*?\$)", text)
#     escaped_parts = []

#     for part in parts:
#         if part.startswith('$') and part.endswith('$'):
#             inner = part[1:-1]

#             # Find any 4+ letter words not part of LaTeX command
#             non_command_words = re.findall(r'(?<!\\)[a-zA-Z]{4,}', inner)
#             if non_command_words:
#                 # Escape whole part (not math)
#                 part = part.replace('\\', r'\\')
#                 part = part.replace('{', r'\{')
#                 part = part.replace('}', r'\}')
#                 part = part.replace('&', r'\&')
#                 part = part.replace('%', r'\%')
#                 part = part.replace('$', r'\$')
#                 part = part.replace('#', r'\#')
#                 part = part.replace('_', r'\_')
#                 part = part.replace('^', r'\^{}')
#                 part = part.replace('~', r'\~{}')
#                 escaped_parts.append(part)
#             else:
#                 escaped_parts.append(part)  # keep math
#         else:
#             part = part.replace('\\', r'\\')
#             part = part.replace('{', r'\{')
#             part = part.replace('}', r'\}')
#             part = part.replace('&', r'\&')
#             part = part.replace('%', r'\%')
#             part = part.replace('$', r'\$')
#             part = part.replace('#', r'\#')
#             part = part.replace('_', r'\_')
#             part = part.replace('^', r'\^{}')
#             part = part.replace('~', r'\~{}')
#             escaped_parts.append(part)

#     return ''.join(escaped_parts)

def escape_latex_text(text):
    # Step 1: Handle $$...$$ blocks (always treat as math)
    def extract_dollar_blocks(text):
        double_math = []
        text_parts = []
        i = 0

        while True:
            match = re.search(r'\$\$(.*?)\$\$', text, flags=re.DOTALL)
            if not match:
                text_parts.append(text)
                break
            pre, math_expr, post = text[:match.start()], match.group(1), text[match.end():]
            text_parts.append(pre)
            double_math.append(math_expr)
            text = post

        return text_parts, double_math

    text_parts, double_math = extract_dollar_blocks(text)

    final_text = ""
    for i, part in enumerate(text_parts):
        # Now handle $...$ inside the non-$$ part
        segments = re.split(r"(\$.*?\$)", part)
        for seg in segments:
            if seg.startswith("$") and seg.endswith("$"):
                inner = seg[1:-1]
                non_command_words = re.findall(r'(?<!\\)[a-zA-Z]{4,}', inner)
                if non_command_words:
                    # Escape entire segment as plain text
                    seg = seg.replace('\\', r'\\')
                    seg = seg.replace('{', r'\{')
                    seg = seg.replace('}', r'\}')
                    seg = seg.replace('&', r'\&')
                    seg = seg.replace('%', r'\%')
                    seg = seg.replace('$', r'\$')
                    seg = seg.replace('#', r'\#')
                    seg = seg.replace('_', r'\_')
                    seg = seg.replace('^', r'\^{}')
                    seg = seg.replace('~', r'\~{}')
                    final_text += seg
                else:
                    final_text += seg  # keep as math
            else:
                # Escape normal text
                seg = seg.replace('\\', r'\\')
                seg = seg.replace('{', r'\{')
                seg = seg.replace('}', r'\}')
                seg = seg.replace('&', r'\&')
                seg = seg.replace('%', r'\%')
                seg = seg.replace('$', r'\$')
                seg = seg.replace('#', r'\#')
                seg = seg.replace('_', r'\_')
                seg = seg.replace('^', r'\^{}')
                seg = seg.replace('~', r'\~{}')
                final_text += seg
        # After every part, add corresponding $$...$$ block if it exists
        if i < len(double_math):
            final_text += f"${double_math[i]}$"

    return final_text

from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, image_path, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size=40, padding=20):
    """
    Render text to a high-resolution image without any LaTeX or resizing.
    """
    # Load font
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Dummy image to calculate size
    dummy_img = Image.new("RGB", (250, 100))
    draw = ImageDraw.Draw(dummy_img)
    
    text_lines = text.split("\n")
    widths, heights = [], []

    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        widths.append(width)
        heights.append(height)

    img_width = max(widths) + 2 * padding
    img_height = sum(heights) + 2 * padding + (len(text_lines) - 1) * 8

    # Final image
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    y = padding
    for line, h in zip(text_lines, heights):
        draw.text((padding, y), line, fill="black", font=font)
        y += h + 8  # Line spacing

    img.save(image_path)


# def render_text_to_image(text, image_path):
#     text = escape_latex_text(text)

#     # Temporary figure to calculate the bounding box
#     fig = plt.figure()
#     text_obj = fig.text(0, 1, text, fontsize=14, wrap=True, ha='left', va='top')

#     # Draw and get bounding box in display coordinates
#     fig.canvas.draw()
#     bbox = text_obj.get_window_extent()

#     # Convert from display to inches
#     dpi = fig.dpi
#     width_in = bbox.width / dpi + 0.5
#     height_in = bbox.height / dpi + 0.5
#     plt.close(fig)

#     # Final figure with correct size
#     fig = plt.figure(figsize=(width_in, height_in))
#     fig.text(0, 1, text, fontsize=14, wrap=True, ha='left', va='top')
#     plt.axis("off")

#     fig.savefig(image_path, bbox_inches='tight', pad_inches=0.3)
#     plt.close(fig)

def render_text_to_image(text, image_path):
    try:
        plt.rcParams.update({
            "text.usetex": True
        })

        # Try escaping + rendering
        escaped_text = escape_latex_text(text)

        # First render pass (to get layout size)
        fig = plt.figure()
        text_obj = fig.text(0, 1, escaped_text, fontsize=14, wrap=True, ha='left', va='top')
        fig.canvas.draw()
        bbox = text_obj.get_window_extent()
        dpi = fig.dpi
        width_in = bbox.width / dpi + 0.5
        height_in = bbox.height / dpi + 0.5
        plt.close(fig)

        # Final render with proper size
        fig = plt.figure(figsize=(width_in, height_in))
        fig.text(0, 1, escaped_text, fontsize=14, wrap=True, ha='left', va='top')
        plt.axis("off")
        fig.savefig(image_path, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)

    except Exception as e:
        text_to_image(text, image_path, font_size=20)
        


# Convert each problem to image
# for idx, example in enumerate(dataset):
#     problem_text = example["problem"]
#     image_path = os.path.join(output_dir, f"problem_{idx:05d}.png")
#     render_text_to_image(problem_text, image_path)
#     print(f"[Saved] {image_path}")


# print(f"✅ Done! Rendered {len(dataset)} math problems with LaTeX to: {output_dir}")
# Add image column
image_paths = []
for idx, example in enumerate(dataset):
    print("example")
    image_path = os.path.join(output_dir, f"problem_{idx:05d}.png")
    render_text_to_image(example["problem"], image_path)
    image_paths.append(image_path)
    print(f"[Saved] {image_path}")

# image_column = [os.path.join(image_dir, f"problem_{idx:05d}.png") for idx in range(len(dataset))]

# Replace image paths with actual image data
from datasets import load_dataset, Image
dataset = dataset.add_column("image", image_paths)
dataset = dataset.cast_column("image", Image())
# Add image column to dataset
# dataset = dataset.add_column("image", image_paths)

# Push to Hugging Face
dataset.push_to_hub("./math_mm_open-r1-math-500")