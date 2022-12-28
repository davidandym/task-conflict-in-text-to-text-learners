import types
import random

import numpy as np
import torch

from datasets import DATASET



def get_task_type(benchmark, task):

    task_type_str = DATASET[benchmark].task_types[task]
    
    task_type = types.SimpleNamespace()

    task_type.name = task_type_str
    task_type.classification = task_type_str == 'classification'
    task_type.regression = task_type_str == 'regression'
    task_type.seq2seq = task_type_str == 'seq2seq'
    task_type.span = task_type_str == 'spanlabeling'
    task_type.span_no = task_type_str == 'spanlabeling_noanswer'
    return task_type


def set_seed(args, rank=None):
    if rank is None and len(args.devices) > 0:
        ordinal = args.devices[0]
    else:
        ordinal = args.devices[rank] 
    device = torch.device(f'cuda:{ordinal}' if ordinal > -1 else 'cpu')
    print(f'device: {device}')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.cuda.device(ordinal):
        torch.cuda.manual_seed(args.seed)
    return device