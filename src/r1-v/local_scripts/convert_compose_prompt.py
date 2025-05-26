import json

# Define a detailed, realistic example
# EXAMPLE = (
#     "Example:\n"
#     "There are five shapes in an 8×8 grid. The target is a red right_triangle at (2, 2).\n"
#     "- Shape_0: red right_triangle at (2, 2), base = 6, height = 4, area = 12.0\n"
#     "- Shape_1: blue square at (3, 2), side = 5, area = 25\n"
#     "- Shape_2: green rectangle at (2, 4), width = 4, height = 10, area = 40\n"
#     "- Shape_3: yellow trapezoid at (6, 7), base1 = 3, base2 = 5, height = 2, area = 8.0\n"
#     "- Shape_4: purple right_triangle at (1, 1), base = 2, height = 3, area = 3.0\n"
#     "Question: What is the total area of the 2 nearest shapes to the red right_triangle? (Round the result to the nearest integer)\n"
#     "Thinking: Target: red right_triangle at (2, 2)\n"
#     "- blue square at (3, 2): distance = |2 - 3| + |2 - 2| = 1, area = 25\n"
#     "- green rectangle at (2, 4): distance = |2 - 2| + |2 - 4| = 2, area = 40\n"
#     "Sum of areas = 65.0 → Rounded = 65\n"
# )
example = "Example:\nIf the target is blue rectangle at (2, 0), the example thinking path and answer is:\n<think>target blue rectangle at (2, 0) has area: $\\text{Rectangle: } w=5, h=7 \\Rightarrow A=w \\times h = 5 \\times 7 = 35$\n- green square at (4, 0): distance = |2 - 4| + |0 - 0| = 2, area: $\\text{Square: } s=2 \\Rightarrow A=s^2=2^2=4$\n- yellow trapezoid at (3, 7): distance = |2 - 3| + |0 - 7| = 8, area: $\\text{Trapezoid: } a=3, b=6, h=4 \\Rightarrow A=\\frac{a+b}2 \\times h = \\frac{3+6}2 \\times 4 = 18.0$. The nearest shape is the green square at (4, 0). Sum of areas (target + nearest) = 35 + 4 = 39 → Rounded = 39</think>\n<answer>39</answer>\n"
instruction = "\nYou need to describe the content of the image first and then answer the question."

def add_detailed_example(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            # if not data["question"].startswith("Example:"):
            # data["question"] = f"{example}{data['question']}"
            data["question"] = f"{instruction}{data['question']}"
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_jsonl = "./src/eval/prompts/shape-spatial-reasoning-eval-simple.jsonl"  # Replace with your actual file path
    output_jsonl = "./src/eval/prompts/shape-spatial-reasoning-eval-simple-describe.jsonl"
    add_detailed_example(input_jsonl, output_jsonl)