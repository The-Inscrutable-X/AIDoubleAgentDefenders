#!/bin/bash
source .env
source "$VENV_PATH"
cd "$REPO_DIR"
set -x

device="$1"
torch_seed_override="$2"
shift 2
config_yamls=("$@")
results_dir="$RESULTS_DIR"
configs=(
    "42|utils/sequence_of_prompt_ablations/prompts_v5_2x"
    "42|utils/sequence_of_prompt_ablations/prompts_v6_2x"
    "42|utils/sequence_of_prompt_ablations/orig_v4p1_2x"
    "42|utils/sequence_of_prompt_ablations/orig_v4_2x"
)

for config_yaml in "${config_yamls[@]}"
do
    for config in "${configs[@]}"
    do
        IFS='|' read -r torch_seed attacker_prompts_dir <<< "$config"
        torch_seed="${torch_seed_override:-$torch_seed}"
        CUDA_VISIBLE_DEVICES=$device python -m main_scripts.main_training_script \
            --config "$config_yaml" \
            --results_dir "$results_dir" \
            --torch_seed $torch_seed \
            --attacker_prompts_dir $attacker_prompts_dir
    done
done
