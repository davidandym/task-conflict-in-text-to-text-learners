#! /usr/bin/env bash

python src/cache_dataset.py \
    --raw_data_dir "/exp/dmueller/data/mqan" \
    --cached_dataset_path "/exp/dmueller/task-specification/camera-ready-tests/.cache/canonical-deca-test.torch" \
    --benchmark "decanlp" \
    --train_tasks "all-tasks" \
    --canonical \
    --prompt_style "canonical" \
