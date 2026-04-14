#!/bin/bash
source .env
cd "$REPO_DIR"
set -x

python datasets_directory/data_generation_scripts/dataset_generation.py \
    --mode theme \
    --num_attempts 10 \
    --output_dir datasets_directory/test_raw_datasets

python datasets_directory/data_generation_scripts/transform_dataset.py \
    --input_dir datasets_directory/test_raw_datasets \
    --output_file datasets_directory/test_transformed_dataset.json
