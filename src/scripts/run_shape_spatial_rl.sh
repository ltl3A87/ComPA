#!/bin/bash


# The latest vllm==0.7.3 is required for this script: pip3 install vllm==0.7.3
# The latest transformers is required too, install by: pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e



export DEBUG_MODE="true"
export LOG_PATH="./qwenvl_25_7B_mix_R1V_Train_8K_rl.txt"


QWEN_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
HF_DATASET="./mm_r1_combined_easy_modify" 
OUTPUT_DIR=".../ComPA/outputs/qwenvl_25_7B_mix_R1V_Train_8K_rl"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="qwenvl_25_7B_mix_R1V_Train_8K_rl"
DS_CONFIG="src/r1-v/local_scripts/zero3.json"

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=7 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/r1-v/src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --use_vllm true \
    --vllm_device "cuda:7" \
    --vllm_gpu_memory_utilization 0.8 \
    --attn_implementation flash_attention_2 \
    --max_prompt_length 4096 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_total_limit 1 \
    --save_only_model true \
    --report_to wandb \
    --temperature 1.0 \
    --num_generations 8 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/${RUN_NAME}"

