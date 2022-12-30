import types
import random

import numpy as np
import torch

from data import DATASET_MAP



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
    device = torch.device('cuda:0')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.cuda.device('cuda:0'):
        torch.cuda.manual_seed(args.seed)
    return device
