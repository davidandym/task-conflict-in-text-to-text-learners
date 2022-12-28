# Dataset class for the GLUE Benchmark.
# Uses Huggingface datasets: https://huggingface.co/datasets/glue



import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from dataset_template import Dataset
from prompt_templates.glue_templates import GlueTemplate



class GlueDataset(Dataset):

    all_tasks = [
        'cola',
        'rte',
        'mnli',
        'mrpc',
        'qnli',
        'qqp',
        'sst2',
        'stsb'
    ]

    task_metric_keys = {
        'sst2': ['accuracy'],
        'mnli': ['accuracy'],
        'mnli_mismatched': ['accuracy'],
        'qnli': ['accuracy'],
        'rte':  ['accuracy'],
        'mrpc': ['accuracy', 'f1'],
        'qqp':  ['accuracy', 'f1'],
        'stsb': ['pearson', 'spearmanr'],
        'cola': ['matthews_correlation']
    }

    task_types = {
        'cola': 'classification',
        'rte': 'classification',
        'mnli': 'classification',
        'mrpc': 'classification',
        'qnli': 'classification',
        'qqp': 'classification',
        'sst2': 'classification',
        'stsb': 'regression'
    }


    def __init__(self, args, tokenizer: PreTrainedTokenizer):

        super.__init__(self, args, tokenizer)

        # prompt templates
        self.prompt_template = GlueTemplate(args)

        # extra eval dataset
        self.dataset["mnli_mismatched"]['test'] = self.load_data("mnli_mismatched", tokenizer, mode='test')
        self.dataset_size["mnli_mismatched"]['test'] = len(self.dataset["mnli_mismatched"]['test'])


    def load_data(self, task, tokenizer, mode='dev'):

        if task == 'mnli' and mode != 'train':
            mode += '_matched'

        hf_dataset = load_dataset('glue', task, split=mode, cache_dir=self.raw_data_dir)

        dev_split = int(len(hf_dataset) * 0.95)
        if mode == "dev":
            hf_dataset = hf_dataset[dev_split:]
        if mode == "train":
            hf_dataset = hf_dataset[:dev_split]

        features = []
        count = 0
        for example in hf_dataset:
            fields = self.gather_fields(task, example, mode)

            if fields is None:
                continue

            s_encoded = tokenizer.encode_plus(
                fields['sentence'],
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                return_attention_mask=True,
                pad_to_max_length=True
            )

            ds_fields = {}
            ds_fields['inputs'] = s_encoded['input_ids']
            ds_fields['attention_mask'] = s_encoded['attention_mask']

            if self.canonical:

                if task == 'stsb':
                    ds_fields['label'] = float(fields['label'])
                else:  
                    ds_fields['label'] = int(fields['label'])

            else:

                label_encoded = tokenizer.encode_plus(
                    fields['label'],
                    max_length=16,
                    pad_to_max_length=True,
                    return_attention_mask=False
                )

                ds_fields['label'] = label_encoded['input_ids']


            if count < self.display_count:
                self.display_example(count, task, mode, fields, tokenizer, s_encoded, label_encoded)

            features.append(ds_fields)
            count += 1

        all_inputs = torch.tensor([f['inputs'] for f in features], dtype=torch.long)
        all_masks = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)

        if self.canonical:
            if task == 'stsb':
                all_labels = torch.tensor([f['label'] for f in features], dtype=torch.float)
            else:
                all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)
        else:
            all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)

        return TensorDataset(all_inputs,
                             all_masks,
                             all_labels)


    def gather_fields(self, task, example, mode):
        # The examples of each task have different field names,
        # so this is a lazy solution to deal with this. Avert your eyes :)

        if task == 'cola':
            s1 = example['sentence']
            s2 = None
            label = example['label']
        elif task == 'rte':
            s1 = example['sentence1']
            s2 = example['sentence2']
            label = example['label']
        elif task == 'mnli' or task == 'mnli_mismatched':
            s1 = example['premise']
            s2 = example['hypothesis']
            label = example['label']
        elif task == 'mrpc':
            s1 = example['sentence1']
            s2 = example['sentence2']
            label = example['label']
        elif task == 'qnli':
            s1 = example['question']
            s2 = example['sentence']
            label = example['label']
        elif task == 'qqp':
            s1 = example['question1']
            s2 = example['question2']
            label = example['label']
        elif task == 'sst2':
            s1 = example['sentence']
            s2 = None
            label = example['label']
        elif task == 'stsb':
            s1 = example['sentence1']
            s2 = example['sentence2']
            label = example['label']
        else:
            raise Exception("Unknown task in data gathering")

        prompted_inputs = self.prompt_template.encode_prompt(task, s1, s2, example, mode)
        if prompted_inputs is None:
            return None

        if self.canonical:
            return {
                'sentence': prompted_inputs,
                'label': label,
            }


        text_outputs = self.prompt_template.encode_label(task, label, example, mode)
        return {
            'sentence': prompted_inputs,
            'label': text_outputs
        }
