""" Dataset Class Template """



from collections import defaultdict

import torch
from transformers import PreTrainedTokenizer



class Dataset():
       
    all_tasks = []
    task_metric_keys = {}

    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        # Saves important parameters and loads datasets.
        # This function can take some time to run, since it tokenizes the entire dataset.
    
        self.tokenizer_type = args.pretrained_model
        
        self.benchmark = args.benchmark
        self.prompt_style = args.prompt_style
        self.label_style = args.label_style
        self.canonical = args.canonical
        self.text_to_text = not self.canonical

        self.display_count = 10 # how many samples to print when loading (for debugging)
        self.max_seq_len = args.max_seq_len
        self.raw_data_dir = args.raw_data_dir
        self.dataset = defaultdict(dict)
        self.dataset_size = defaultdict(dict)

        for task in args.train_tasks:
            self.dataset[task]['test'] = self.load_data(task, tokenizer, mode='test')
            self.dataset_size[task]['test'] = len(self.dataset[task]['test'])

            self.dataset[task]['dev'] = self.load_data(task, tokenizer, mode='dev')
            self.dataset_size[task]['dev'] = len(self.dataset[task]['dev'])

            self.dataset[task]['train'] = self.load_data(task, tokenizer, mode='train')
            self.dataset_size[task]['train'] = len(self.dataset[task]['train'])


    def load_data(self, task, tokenizer, mode='dev'):
        # Override this function in subclass.

        pass


    def sample_train_batch(self, task, batch_size):
        # Simple dataset sampler with replacement.

        train_set = self.dataset[task]['train']
        idcs = torch.randperm(len(train_set))[:batch_size]
        return train_set[idcs]


    def batch_to_inputs(self, batch, task):
        # Convert batched inputs from dataset to input dict for torch model.

        inputs = {}
        inputs["input_ids"] = batch[0]
        inputs["attention_mask"] = batch[1]
        inputs["labels"] = batch[2]

        return inputs


    def display_example(self, count, task, mode, fields, tokenizer,
                        s_encoded):
        print(f"{'<'*14} Example {count} for {task} {mode} {'>'*14}")

        print("<<<< tokenized input example")
        print(tokenizer.tokenize(fields['sentence'],
                                add_special_tokens=True))
        print(s_encoded["input_ids"])

        print("<<<<< Label")
        print(fields['label'])


    def check_consistency(self, args):
        # A method to check for consistency between a dataset and a set of args.
        # Used for ensuring that pre-cached datasets match the experiment they are
        # loaded for.

        consistency = True
        failures = []

        # Check benchmark consistency
        if self.benchmark != args.benchmark:
            consistency = False
            failures.append('benchmark')


        # Check task consistency
        for task in args.train_tasks:
            if task not in self.dataset:
                consistency = False
                failures.append(f'task {task}')
        
        if self.canonical != args.canonical:
            consistency = False
            failures.append('canonical')

        # Check prompt consistency
        if self.prompt_style != args.prompt_style:
            consistency = False
            failures.append('prompt style')

        # Check label consistency only if we're in the text-to-text setting
        if not self.canonical:
            if self.label_style != args.label_style:
                consistency = False
                failures.append('label style')
        
        # Check max sequence length consistency.
        if self.max_seq_len != args.max_seq_len:
            consistency = False
            failures.append('max seq len')


        # Check max sequence length consistency.
        if self.tokenizer_type != args.pretrained_model:
            consistency = False
            failures.append('pretrained tokenizer')

        return (consistency, failures)

