#!/usr/bin/env python3



import os
import csv
import time
import collections
import argparse as ap
import logging
from pprint import pformat
from logging import handlers

import ujson as json
import torch
from transformers import AutoConfig, T5Tokenizer
from transformers import AdamW, get_scheduler

from datasets import DATASET
from models import CanonicalT5, Text2TextT5, Text2TextIndependentHeadsT5
from utils import set_seed
from evaluate import evaluate
from measure_conflict import measure_conflict, get_conflict_metric_keys



def get_args():

    p = ap.ArgumentParser()

    # IO Parameters
    p.add_argument("--log_dir", default="experiments/t5/testing",
                   help="output directory")
    p.add_argument("--cached_dataset_path", default="/exp/dmueller/data/glue/.cached_t5_project_dataset.torch",
                   help="location of pre-trained dataset")
    p.add_argument("--log_every", default=100, type=int,
                   help="Num of steps between logging of training.")

    # Core Experiment Parameters
    p.add_argument("--benchmark", default='glue', type=str,
                  help="Either glue or decanlp")
    p.add_argument("--train_tasks", default=['cola'], nargs='+',
                  help="Tasks to train on. 'all-tasks' to include all tasks in a benchmark.")
    p.add_argument('--canonical', action='store_true',
                    help="set if using a canonical model and output spaces.")
    p.add_argument('--text-to-text', dest='canonical', action='store_false',
                    help="set if using a text-to-text model and output spaces.")
    p.add_argument('--t2t-independent-heads', action='store_true',
                    help="set if using independent heads with a text-to-text model.")
    p.add_argument("--reset_decoder_weights", action="store_true",
                   help="Reset the weights of the T5 Decoder for text-to-text models.")
    p.add_argument('--prompted_style', default='canonical', type=str,
                    help="The style of prompts to use for the model. Applies to canonical & text-to-text experiments.")
    p.add_argument('--label_style', default='default', type=str,
                    help="The style of label to use for the model. Only applies to text-to-text experiments.")
    p.set_defaults(canonical=True,)

    # T5
    p.add_argument("--pretrained_model", default="google/t5-v1_1-small",
                   help="pre-trained T5 model type")

    # Validation Settings
    p.add_argument("--val_every", default=500, type=int,
                   help="Num of steps between evaluations.")
    p.add_argument("--val_batch_size", default=32, type=int,
                   help="Num examples per validation fwd pass.")

    # Conflict Measurement Parameters
    p.add_argument("--measure_conflict_decoder", action="store_true",
                   help="Collect decoder gradients")
    p.add_argument("--measure_conflict_every", default=500, type=int,
                   help="Num of steps between conflict measurements.")
    p.add_argument("--conflict_large_batch_size", default=1024, type=int,
                   help="Num examples with which to measure conflict.")
    p.add_argument("--conflict_small_batch_size", default=16, type=int,
                   help="Small batch size for estimating task-noise.")
    p.add_argument("--measure_conflict_on_cpu", action="store_true",
                   help="Store conflict gradients on the cpu.")

    # Optimization Hyperparameters
    p.add_argument("--grad_accum", default="sum", type=str,
                   help="How to accumulate multi-task gradients (sum or avg).")
    p.add_argument("--grad_clip", default=5.0, type=float,
                   help="Maximum gradient norm.")
    p.add_argument("--dropout_rate", default=0.0, type=float,
                   help="Dropout rate for models.")
    p.add_argument("--train_batch_size", default=32, type=int,
                   help="Base learning rate for optimizer.")
    p.add_argument("--lr", default=1e-3, type=float,
                   help="Base learning rate for optimizer.")
    p.add_argument("--learning_rate_schedule", default="cosine",
                   help="Base learning rate for optimizer.")
    p.add_argument("--max_steps", default=10000, type=int,
                   help="Number of total steps.")
    p.add_argument("--warmup_steps", default=500, type=int,
                   help="Number of steps to warmup, if applicable.")
    p.add_argument("--seed", default=1, type=int,
                  help="random seed for experiment")

    return p.parse_args()



def initialize_logger(args, rank='main'):
    # set up file logger

    logger = logging.getLogger(f'process_{rank}')
    logger.setLevel(logging.DEBUG)
    handler = handlers.RotatingFileHandler(os.path.join(args.log_dir, f'process_{rank}.log'), maxBytes=1024*1024*10, backupCount=1)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger



def initialize_csv_logger(args, dataset):
    # This is the main logging mechanism I use to record training trajectory information for this project.
    # Initializing a CSV writer in this way requires knowing which columns will be present before hand, so
    # I've written some helper functions to keep note of what columns need to be declared here.

    train_log_fields = ['step', 'total train loss', 'total dev score',
                        'inter-task directional conflict',
                        'inter-task magnitude conflict']
    train_log_fields += [f'{task} train loss' for task in args.train_tasks]
    for task in args.train_tasks:
        metrics = dataset.task_metric_keys[task]
        train_log_fields += [f'{task} dev {metric}' for metric in metrics]
        train_log_fields += [f'{task} dev total samples']
        train_log_fields += [f'{task} dev loss']
        train_log_fields += [f'{task} dev bad outputs']
    train_log_fields += get_conflict_metric_keys(args.train_tasks)

    train_log_file = open(os.path.join(args.log_dir, 'train_logs.csv'), 'w', newline='')
    train_logger = csv.DictWriter(
        train_log_file,
        fieldnames=train_log_fields
    )
    train_logger.writeheader()

    return train_logger, train_log_file



def log(rank='main'):
    return logging.getLogger(f'process_{rank}')



def apply_grads(model, gradients, num_tasks, accumulation='sum'):

    for n, p in model.named_parameters():
        if n in gradients:
            if 'task_specific_classifiers' in n:
                grad = gradients[n].view(p.shape)
                p.grad = grad
            else:
                if accumulation == 'sum':
                    grad = gradients[n].view(p.shape)
                    p.grad = grad
                elif accumulation == 'avg':
                    # I don't really recommend using avg, summing task gradients generally results in better generalization.
                    # See https://openreview.net/forum?id=H9UOWMR_Ut ;)
                    grad = gradients[n].view(p.shape)
                    p.grad = grad.mul(1/num_tasks)
                else:
                    raise Exception("Gradient accumulation method {} not recognized".format(accumulation))



def collect_grads(gradients, model, store_device='cuda'):
    for n, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if n not in gradients:
            gradients[n] = torch.unsqueeze(torch.squeeze(p.grad.clone()), dim=0)
        else:
            gradients[n] += torch.unsqueeze(torch.squeeze(p.grad.clone()), dim=0)
    return gradients



def train(args, device,
          model, dataset, tokenizer, opt, lr_scheduler,
          train_logger, train_log_file):
    # The core training loop.

    max_steps = args.max_steps
    iteration = 0

    while True:
        step_log = {'step': iteration}

        # For storing intermediate task-specific gradients.
        gradients = {}

        total_train_loss = 0.
        for task in args.train_tasks:
            # Collect task-specific gradients.

            model.train()
            opt.zero_grad()

            batch = dataset.sample_train_batch(task, args.train_batch_size)
            batch = tuple(t.to(device) for t in batch)
            inputs = dataset.batch_to_inputs(batch, task, args.canonical, args.prompted_inputs)

            out = model(task, tokenizer, **inputs)

            loss = out.loss

            step_log[f'{task} train loss'] = float(loss)
            total_train_loss += float(loss)

            loss.backward()
            gradients = collect_grads(gradients, model)

        step_log['total train loss'] = total_train_loss
        opt.zero_grad()

        # Aggregate all task gradients into a single gradient.
        apply_grads(model, gradients, len(args.train_tasks), accumulation=args.grad_accum)

        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Take a step in the aggregated direction.
        opt.step()
        lr_scheduler.step()

        # Validate.
        if iteration % args.val_every == 0:
            for task in args.train_tasks:

                metric_dict = evaluate(args, device, task, model, dataset, tokenizer)
                for metric_key, metric_value in metric_dict.items():
                    step_log[f'{task} dev {metric_key}'] = metric_value

        # measure conflict
        if args.measure_conflict_every > 0 and iteration % args.measure_conflict_every == 0:
            conflict_metrics = measure_conflict(args,
                                                args.train_tasks,
                                                model,
                                                tokenizer,
                                                opt,
                                                device,
                                                dataset,
                                                args.conflict_large_batch_size,
                                                args.conflict_small_batch_size,
                                                args.measure_conflict_decoder)
            step_log.update(conflict_metrics)

        # Log this step.
        if iteration % args.log_every == 0:
            train_logger.writerow(step_log)
            train_log_file.flush()

        # completed one iteration loop
        iteration += 1

        # Break condition
        if iteration > max_steps:
            train_log_file.close()
            break



def evaluate_model(args, device, model, dataset, tokenizer, eval_log_file):
    # Run final evaluation of the model on dev and test sets.

    evaluation = collections.defaultdict(dict)

    for task in args.train_tasks:
        metric_dict = evaluate(args, device, task, model, dataset, tokenizer)
        evaluation['dev'][task] = metric_dict

        metric_dict = evaluate(args, device, task, model, dataset, tokenizer, test=True)
        evaluation['test'][task] = metric_dict

    if 'mnli' in args.train_tasks:
        # If we're in GLUE and we trained on MNLI, we also evaluate MNLI-MM test set.
        metric_dict = evaluate(args, device, 'mnli_mismatched', model, dataset, tokenizer, task_head='mnli', test=True)
        evaluation['test']['mnli_mismatched'] = metric_dict

    json.dump(evaluation, eval_log_file)



def run(args, device, model, dataset, tokenizer):

    logger = initialize_logger(args)
    logger.start = time.time()

    logger.info(f'Preparing iterators')

    opt, lr_scheduler = init_opt(args, model)

    train_logger, train_log_file = initialize_csv_logger(args, dataset)
 
    # Experiment Starts Here.
    train(args, device,
          model, dataset, tokenizer, opt, lr_scheduler,
          train_logger, train_log_file)

    eval_log_file = open(os.path.join(args.log_dir, 'evaluation.json'), 'w')
    evaluate_model(args, device, model, dataset, tokenizer, eval_log_file)
    eval_log_file.close()
    # Experiment Finishes Here.



def init_model(args, config, device):
    # Load model based on experiment settings.

    if args.canonical:
        # If canonical is true, load the canonical multi-task model.
        model = CanonicalT5(args, config).cuda(device)
    else:
        # Otherwise, the model is a text-to-text multi-task model.
        if args.t2t_independent_heads:
            # Independent heads Text-to-Text model.
            model = Text2TextIndependentHeadsT5(args, config=config).cuda(device)
            if args.reset_decoder_weights:
                for task in args.train_tasks:
                    model.task_specific_heads[task].decoder.init_weights()
        else:
            # Fully unified Text-to-Text model.
            model = Text2TextT5.from_pretrained(args.pretrained_model, config=config).cuda(device)
            if args.reset_decoder_weights:
                model.decoder.init_weights()

    return model



def init_opt(args, model):
    # Inialize the optimizer and learning rate scheduler.

    opt = AdamW(model.parameters(), lr=args.lr)
    lr_sched = get_scheduler(args.learning_rate_schedule,
                             opt,
                             num_warmup_steps=args.warmup_steps,
                             num_training_steps=args.max_steps)
    return opt, lr_sched



def main():

    args = get_args()
    if args is None:
        return

    # Clean up args.
    args.benchmark = args.benchmark.lower()
    if args.train_tasks == ['all-tasks']:
        args.train_tasks = DATASET[args.benchmark].all_tasks

    # Initialize seed and args.
    device = set_seed(args)
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    # Load huggingface T5 Tokenizer.
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model)

    # Because dataset loading takes so long, I recommend cacheing it first.
    # dataset = DATASET[args.benchmark](args)
    dataset = torch.load(args.cached_dataset_path)
    consistent, failures = dataset.check_consistency(args)
    if not consistent:
        raise Exception(f"The loaded dataset is not consistent with the following args: {' '.join(failures)}")

    # Load huggingface T5 Config
    config = AutoConfig.from_pretrained(args.pretrained_model)
    # In my experiments, I found that setting dropout rate to 0. was very important.
    config.update({"dropout_rate": args.dropout_rate})

    # Load model architecture.
    model = init_model(args, device)

    # Run experiment.
    run(args, device, model, dataset, tokenizer)



if __name__ == '__main__':
    main()
