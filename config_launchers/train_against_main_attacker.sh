#!/bin/bash
source .env
source "$VENV_PATH"
cd "$REPO_DIR"
set -x

device="$1"
config_yaml="$2"
torch_seed_override="$3"
results_dir="$RESULTS_DIR"
configs=(
    "4|dr_grpo|42|utils/sequence_of_prompt_ablations/orig_v4_2x|0.0"
)

for config in "${configs[@]}"
do
    IFS='|' read -r gradient_accumulation_steps loss_type torch_seed attacker_prompts_dir warmup_ratio <<< "$config"
    torch_seed="${torch_seed_override:-$torch_seed}"
    CUDA_VISIBLE_DEVICES=$device python -m main_scripts.main_training_script \
        --config "$config_yaml" \
        --results_dir "$results_dir" \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --loss_type $loss_type \
        --torch_seed $torch_seed \
        --attacker_prompts_dir $attacker_prompts_dir \
        --lr_warmup_ratio $warmup_ratio
done
