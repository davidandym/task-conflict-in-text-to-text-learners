#!/usr/bin/env python3



import argparse as ap

import torch
from transformers import T5Tokenizer

from data import DATASET_MAP



def get_args():

    p = ap.ArgumentParser()

    # IO Parameters
    p.add_argument("--cached_dataset_path", default="/exp/dmueller/data/glue/.cached_t5_project_dataset.torch",
                   help="location of pre-trained dataset")
    p.add_argument("--raw_data_dir", default="/exp/dmueller/data/glue/.cached_t5_project_dataset.torch",
                   help="Location of raw datafiles. (Can be left blank for GLUE to use default huggingface cache directory)")

    # Core Experiment Parameters
    p.add_argument("--benchmark", default='glue', type=str,
                  help="Either glue or decanlp")
    p.add_argument("--train_tasks", default=['cola'], nargs='+',
                  help="Tasks to train on. 'all-tasks' to include all tasks in a benchmark.")
    p.add_argument('--canonical', action='store_true',
                    help="set if using a canonical model and output spaces.")
    p.add_argument('--text-to-text', dest='canonical', action='store_false',
                    help="set if using a text-to-text model and output spaces.")
    p.add_argument('--prompt_style', default='canonical', type=str,
                    help="The style of prompts to use for the model. Applies to canonical & text-to-text experiments.")
    p.add_argument('--label_style', default='default', type=str,
                    help="The style of label to use for the model. Only applies to text-to-text experiments.")
    p.set_defaults(canonical=True,)

    # T5
    p.add_argument("--pretrained_model", default="google/t5-v1_1-small",
                   help="pre-trained T5 model type")
    p.add_argument("--max_seq_len", default=256, type=int,
                   help="Maximum input sequence length.")

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.train_tasks[0] == "all-tasks":
        args.train_tasks = DATASET_MAP[args.benchmark].all_tasks

    print("Loading Tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model)

    print("Loading Dataset")
    dataset = DATASET_MAP[args.benchmark](args, tokenizer)

    print(f"Done Loading Dataset, Saving at {args.cached_dataset_path}")
    torch.save(dataset, args.cached_dataset_path)
