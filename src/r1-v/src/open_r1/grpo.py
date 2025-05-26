# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

import sys
import os

from typing import List, Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    format_caption: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to format the caption"},
    )
    progress: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use progress reward"},
    )


def extract_think_block(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_area_values(text: str) -> List[float]:
    """Extract only area results (numbers after =) from the reasoning block."""
    matches = re.findall(r"=\s*([0-9]+\.?[0-9]*)", text)
    return [float(m) for m in matches]


def extract_position(text: str) -> List[Tuple[int, int]]:
    return [tuple(map(int, match)) for match in re.findall(r"\((\d+),\s*(\d+)\)", text)]


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def determine_problem_type_from_solution(sol: str) -> str:
    think_block = extract_think_block(sol)
    return "grid" if think_block.lower().strip().startswith("target") else "area"


def extract_meta_from_solution(sol: str) -> Dict:
    think = extract_think_block(sol)
    meta = {}
    if not think:
        return meta

    problem_type = determine_problem_type_from_solution(sol)
    if problem_type == "area":
        meta["type"] = "area"
        meta["areas"] = extract_area_values(think)
    elif problem_type == "grid":
        meta["type"] = "grid"
        lines = think.splitlines()
        positions = []
        distances = {}
        for line in lines:
            pos_match = re.search(r"\((\d+),\s*(\d+)\)", line)
            dist_match = re.search(r"distance\s*=\s*(\d+)", line)
            if pos_match:
                pos = (int(pos_match.group(1)), int(pos_match.group(2)))
                positions.append(pos)
                if dist_match:
                    distances[pos] = int(dist_match.group(1))
        if positions:
            meta["target"] = positions[0]
            meta["candidates"] = positions[1:]
            meta["distances"] = {pos: distances.get(pos, manhattan_distance(positions[0], pos)) for pos in positions[1:]}
    return meta


def extract_all_metas(solutions: List[str]) -> List[Dict]:
    return [extract_meta_from_solution(sol) for sol in solutions]


# === Progress reward functions ===

def progress_reward_area(think_block: str, ground_truth_areas: List[float]) -> float:
    predicted_areas = extract_area_values(think_block)
    matched = 0
    used = set()
    for truth in ground_truth_areas:
        for i, pred in enumerate(predicted_areas):
            if i in used:
                continue
            if abs(pred - truth) < 1e-2:
                matched += 1
                used.add(i)
                break
    return matched / len(ground_truth_areas) if ground_truth_areas else 0.0


def progress_reward_grid(think_block: str, distances: Dict[Tuple[int, int], int]) -> float:
    lines = think_block.splitlines()
    matched = 0
    for pos, dist in distances.items():
        pattern = rf"\({pos[0]}\s*,\s*{pos[1]}\).*?{dist}\b"
        if any(re.search(pattern, line) for line in lines):
            matched += 1
    return matched / len(distances) if distances else 0.0


def progress_reward_dispatch(content: str, solution_meta: Dict, problem_type: str) -> float:
    think_block = extract_think_block(content)
    if problem_type == "area":
        return progress_reward_area(think_block, solution_meta.get("areas", []))
    elif problem_type == "grid":
        return progress_reward_grid(think_block, solution_meta.get("distances", {}))
    return 0.0


# === Main reward function ===

# def accuracy_reward(completions, solution, metas=None, **kwargs):
#     """Reward function with accuracy and progress supervision."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

#     for i, (content, sol) in enumerate(zip(contents, solution)):
#         reward = 0.0
#         progress = 0.0

#         try:
#             sol_match = re.search(r'<answer>(.*?)</answer>', sol)
#             ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
#             content_match = re.search(r'<answer>(.*?)</answer>', content)
#             student_answer = content_match.group(1).strip() if content_match else content.strip()

#             # Accuracy check
#             if student_answer == ground_truth:
#                 reward = 1.0
#             else:
#                 try:
#                     if float(student_answer) == float(ground_truth):
#                         reward = 1.0
#                 except Exception:
#                     pass

#             # Extract progress reward
#             meta = metas[i] if metas else extract_meta_from_solution(sol)
#             print("meta: ", meta)
#             problem_type = meta.get("type", determine_problem_type_from_solution(sol))
#             progress = progress_reward_dispatch(content, meta, problem_type)

#         except Exception:
#             pass

#         total_reward = reward + progress
#         rewards.append(total_reward)

#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH", "reward_debug.log")
#             with open(log_path, "a", encoding="utf-8") as f:
#                 f.write(f"--- {current_time} ---\n")
#                 f.write(f"Content: {content}\n")
#                 f.write(f"Solution: {sol}\n")
#                 f.write(f"Accuracy: {reward}, Progress: {progress}, Total: {total_reward}\n")

#     return rewards

def progress_reward(completions, solution, metas=None, **kwargs):
    """Reward function with accuracy and progress supervision."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for i, (content, sol) in enumerate(zip(contents, solution)):
        progress = 0.0

        try:
            # Extract progress reward
            meta = metas[i] if metas else extract_meta_from_solution(sol)
            print("meta: ", meta)
            problem_type = meta.get("type", determine_problem_type_from_solution(sol))
            progress = progress_reward_dispatch(content, meta, problem_type)

        except Exception:
            pass

        rewards.append(progress)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "reward_debug.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"--- {current_time} ---\n")
                f.write(f"Progress Content: {content}\n")
                f.write(f"Progress Solution: {sol}\n")
                f.write(f"Progress reward: {progress}")

    return rewards


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            if ground_truth.startswith("("):
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                if student_answer == ground_truth:
                    reward = 1.0
                    print(f"reward == 1.0 for spatial")
                    print(f"Correct answer: {content}")
                    print(f"Ground truth: {ground_truth}")
            else:
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                answer = parse(student_answer)
                if float(verify(answer, parse(ground_truth))) > 0:
                    reward = 1.0
                if reward == 0.0:
                    try:
                        # Extract answer from solution if it has think/answer tags
                        # sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                        # ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract answer from content if it has think/answer tags
                        # content_match = re.search(r'<answer>(.*?)</answer>', content)
                        # student_answer = content_match.group(1).strip() if content_match else content.strip()
                        
                        # Compare the extracted answers
                        if student_answer == ground_truth:
                            reward = 1.0
                    except Exception:
                        pass  # Keep reward as 0.0 if both methods fail
                if reward == 1.0:
                    print(f"reward == 1.0 for shape")
                    print(f"Correct content: {content}")
                    print(f"answer: {answer}")
                    print(f"Ground truth: {ground_truth}")
                    # print(f"sol: {sol}")
                    print(f"parse(sol): {parse(ground_truth)}")
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward_caption(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

reward_funcs_caption_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward_caption,
}

reward_funcs_progress_registry = {
    "accuracy": accuracy_reward,
    "progress": progress_reward,
    "format": format_reward,
}

reward_funcs_caption_progress_registry = {
    "accuracy": accuracy_reward,
    "progress": progress_reward,
    "format": format_reward_caption,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think>reasoning process here</think><answer>answer here</answer>"
)


def main(script_args, training_args, model_args):
    import torch
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Set device to local_rank {local_rank}")
    # Get reward functions
    if script_args.format_caption:
        if script_args.progress:
            reward_funcs = [reward_funcs_caption_progress_registry[func] for func in script_args.reward_funcs]
        else:
            reward_funcs = [reward_funcs_caption_registry[func] for func in script_args.reward_funcs]
    else:
        if script_args.progress:
            reward_funcs = [reward_funcs_progress_registry[func] for func in script_args.reward_funcs]
        else:
            reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }


    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."

    def make_conversation_image(example):
        print(f"[DEBUG] Mapping example: {example['problem']}")
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example['problem'])},
                    ],
                },
            ],
        }
    
    QUESTION_TEMPLATE_CAP = "{Question} Output the description of the image in <caption> </caption>, the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    
    def make_conversation_image_cap(example):
        print(f"[DEBUG] Mapping example: {example['problem']}")
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE_CAP.format(Question=example['problem'])},
                    ],
                },
            ],
        }


    print("dataset[script_args.dataset_train_split].features: ", dataset[script_args.dataset_train_split].features)
    if 'image' in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        if script_args.format_caption:
            dataset = dataset.map(make_conversation_image_cap)
        else:
            dataset = dataset.map(make_conversation_image) # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        # dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    print("RANK:", os.environ.get("RANK"))
    print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
