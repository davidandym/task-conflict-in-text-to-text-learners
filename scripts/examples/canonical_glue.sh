#! /usr/bin/env bash

python src/train.py \
    --seed 1234 \
    --canonical \
    --prompt_style "canonical" \
    --raw_data_dir "/exp/dmueller/data/glue" \
    --log_dir "/exp/dmueller/task-specification/camera-ready-tests/canonical-glue-test" \
    --benchmark "glue" \
    --train_tasks "all-tasks" \
    --log_every 25 \
    --val_every 200 \
    --measure_conflict_every 200 \
    --measure_conflict_on_cpu \
    --dropout_rate 0.0 \
    --grad_accum "sum" \
    --warmup_steps 2000 \
    --train_batch_size 16 \
    --lr "5e-4" \
    --max_steps 20000
