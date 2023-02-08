#! /usr/bin/env bash

python src/train.py \
    --seed 1234 \
    --text-to-text \
    --prompt_style "default" \
    --label_style "default" \
    --log_dir "/exp/dmueller/task-specification/camera-ready-tests/text-to-text-deca-test" \
    --cached_dataset_path "/exp/dmueller/task-specification/camera-ready-tests/.cache/default-text-to-text-deca-test.torch" \
    --benchmark "decanlp" \
    --train_tasks "all-tasks" \
    --log_every 5 \
    --val_every 10 \
    --measure_conflict_every 10 \
    --measure_conflict_on_cpu \
    --dropout_rate 0.0 \
    --grad_accum "sum" \
    --warmup_steps 10 \
    --train_batch_size 16 \
    --lr "5e-4" \
    --max_steps 21