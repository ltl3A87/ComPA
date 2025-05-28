r1_v_path=.../ComPA
cd ${r1_v_path}

model_path=${r1_v_path}/outputs/qwenvl_25_7B_mix_R1V_Train_8K_rl
batch_size=16
output_path=${r1_v_path}/outputs/qwenvl_25_7B_mix_R1V_Train_8K_rl_spatial_reasoning_eval.json
prompt_path=${r1_v_path}/src/eval/prompts/spatial-reasoning-eval.jsonl
gpu_ids=0,1,2,3,4,5,6,7

python src/eval/test_qwen2vl_metamath_multigpu.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids}

output_path=${r1_v_path}/outputs/qwenvl_25_7B_mix_R1V_Train_8K_rl_geometric_reasoning_eval.json
prompt_path=${r1_v_path}/src/eval/prompts/geometric-reasoning-eval.jsonl

python src/eval/test_qwen2vl_metamath_multigpu.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids}

output_path=${r1_v_path}/outputs/qwenvl_25_7B_mix_R1V_Train_8K_rl_geometric+spatial_reasoning_eval.json
prompt_path=${r1_v_path}/src/eval/prompts/geometric-spatial-reasoning-eval.jsonl

python src/eval/test_qwen2vl_metamath_multigpu.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --prompt_path ${prompt_path} \
    --gpu_ids ${gpu_ids}


