#! /usr/bin/env bash

python src/cache_dataset.py \
    --raw_data_dir "/exp/dmueller/data/glue" \
    --cached_dataset_path "/exp/dmueller/task-specification/camera-ready-tests/.cache/canonical-glue-test.torch" \
    --benchmark "glue" \
    --train_tasks "all-tasks" \
    --canonical \
    --prompt_style "canonical" \
