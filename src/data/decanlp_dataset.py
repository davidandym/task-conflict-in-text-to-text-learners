# Dataset class for DecaNLP Benchmark.
# Unlike GLUE, the assumption here is that a lot of this data is stored locally.
# See https://github.com/salesforce/decaNLP for details on downloading and formatting data.
# I copied them, and their data-loading processes. My only changes are largely fitting it into
# the generic dataset class (and adding canonical versions of input / output).



import os
import io
import csv
import json

import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer

from data.dataset_template import Dataset
from prompt_templates.decanlp_templates import DecaNLPTemplate



class DecaNlpDataset(Dataset):

    all_tasks = [
        "squad",
        "iwslt",
        "cnn_dailymail",
        "multinli",
        "sst",
        "srl",
        "zre",
        "schema",
        "wikisql",
    ]

    canonical_output_space = {
        "iwslt":            'text',
        "cnn_dailymail":    'text',
        "wikisql":          'text',
        "multinli":         'class',
        "sst":              'class',
        "srl":              'span',
        "zre":              'span',
        "schema":           'span',
        "squad":            'span',
    }

    task_types = {
        "iwslt":            'seq2seq',
        "cnn_dailymail":    'seq2seq',
        "wikisql":          'seq2seq',
        "multinli":         'classification',
        "sst":              'classification',
        "srl":              'spanlabeling',
        "zre":              'spanlabeling_noanswer',
        "schema":           'spanlabeling',
        "squad":            'spanlabeling',
    }

    
    def __init__(self, args, tokenizer: PreTrainedTokenizer):

        # prompt templates
        self.prompt_template = DecaNLPTemplate(args)

        super().__init__(args, tokenizer)

    
    def load_data(self, task, tokenizer, mode='dev'):
        # All of these tasks have a unique loading functions, for which I am very sorry.
        # The actual loading & processing code itself comes from:
        # https://github.com/salesforce/decaNLP

        if task == 'squad':
            return self.load_squad(tokenizer, mode)
        elif task == 'sst':
            return self.load_sst(tokenizer, mode)
        elif task == 'srl':
            return self.load_srl(tokenizer, mode)
        elif task == 'cnn_dailymail':
            return self.load_cnn_dailymail(tokenizer, mode)
        elif task == 'zre':
            return self.load_zre(tokenizer, mode)
        elif task == 'schema':
            return self.load_schema(tokenizer, mode)
        elif task == 'wikisql':
            return self.load_wikisql(tokenizer, mode)
        elif task == 'multinli':
            return self.load_multinli(tokenizer, mode)
        elif task == 'iwslt':
            return self.load_iwslt(tokenizer, mode)


    def sample_train_batch(self, task, batch_size):
        train_set = self.dataset[task]['train']
        idcs = torch.randperm(len(train_set))[:batch_size]
        return train_set[idcs]


    def batch_to_inputs(self, batch, task):

        inputs = {}
        inputs["input_ids"] = batch[0]
        inputs["attention_mask"] = batch[1]
        if task == 'wikisql':
            inputs['wiki_id'] = batch[3]
        if self.text_to_text:
            inputs["labels"] = batch[2]
            return inputs

        # Otherwise, canonical outputs.
        inputs['labels'] = None
        inputs['labels_mask'] = None
        inputs['labels_span_start'] = None
        inputs['labels_span_end'] = None
        if self.canonical_output_space[task] == 'class':
            inputs['labels'] = batch[2]
        elif self.canonical_output_space[task] == 'text':
            inputs['labels'] = batch[2]
        elif self.canonical_output_space[task] == 'span':
            inputs['labels_span_start'] = batch[2]
            inputs['labels_span_end'] = batch[3]
        else:
            raise Exception("unidentified output space")

        return inputs


   # Really gross processing and loading code below!!


    def load_sst(self, tokenizer, mode):

        features = []

        path = os.path.join(self.raw_data_dir, 'sst', '{}_binary_sent.csv'.format(mode))
        with io.open(os.path.expanduser(path), encoding='utf8') as f:
            next(f) # skip header
            for line in f:
                parsed = list(csv.reader([line.rstrip('\n')]))[0]

                # canonical
                context = parsed[-1]
                label = int(parsed[0])

                # text-to-text
                input = self.prompt_template.encode_prompt('sst', context)
                text_label = self.prompt_template.encode_label('sst', label)

                fields = self.encode(
                    tokenizer=tokenizer,
                    input=input,
                    label=text_label if self.text_to_text else None 
                )

                if self.canonical:
                    fields['label'] = label

                features.append(fields)

        return self.convert_features_to_tensor_dataset(features,
                                                       can_label_class=self.canonical)


    def load_iwslt(self, tokenizer, mode):
        if mode == 'train':
            src_fname = f'train.en-de.en'
            tgt_fname = f'train.en-de.de'
        elif mode == 'dev':
            src_fname = f'IWSLT16.TED.tst2013.en-de.en'
            tgt_fname = f'IWSLT16.TED.tst2013.en-de.de'
        else:
            src_fname = f'IWSLT16.TED.tst2014.en-de.en'
            tgt_fname = f'IWSLT16.TED.tst2014.en-de.de'

        src_path = os.path.join(self.raw_data_dir, 'iwslt', 'en-de', src_fname)
        tgt_path = os.path.join(self.raw_data_dir, 'iwslt', 'en-de', tgt_fname)

        features = []
        with open(src_path) as src_file, open(tgt_path) as tgt_file:
            for src_line, tgt_line in zip(src_file, tgt_file):
                src_line, tgt_line = src_line.strip(), tgt_line.strip()
                if src_line != '' and tgt_line != '':
                    input = src_line
                    input = self.prompt_template.encode_prompt('iwslt', input)
                    label = tgt_line

                    fields = self.encode(
                        tokenizer=tokenizer,
                        input=input,
                        label=label
                    )

                    features.append(fields)

        return self.convert_features_to_tensor_dataset(features, can_label_text=self.canonical)


    def load_squad(self, tokenizer, mode):
        extension = 'v1.1.json'
        train_split = False
        if mode == 'train':
            train_split = True
        if mode == 'dev' or mode == 'train':
            mode = 'train'
        elif mode == 'test':
            mode = 'dev'
        path = os.path.join(self.raw_data_dir, 'squad', f'{mode}-{extension}')

        all_squad = []

        with open(os.path.expanduser(path)) as f:
            # Before tokenization or parsing, we extract all squad data
            squad = json.load(f)['data']

            for document in squad:
                paragraphs = document['paragraphs']

                for paragraph in paragraphs:
                    context = paragraph['context']
                    qas = paragraph['qas']
    
                    for qa in qas:
                        question = ' '.join(qa['question'].split())
                        context_question = self.prompt_template.encode_prompt('squad', context, question)
                        answer = qa['answers'][0]['text']
                
                        all_squad.append((context,
                                          question,
                                          context_question,
                                          answer))

        if mode == 'train' and train_split:
            split = int(len(all_squad) * 0.95)
            all_squad = all_squad[:split]
        elif mode == 'train' and not train_split:
            split = int(len(all_squad) * 0.95)
            all_squad = all_squad[split:]

        features = []

        for squad in all_squad:
            # Now we parse and tokenize, etc.

            context, question, context_question, answer = squad
            context_question = self.clean_punc(context_question)
            answer = self.clean_punc(answer)

            input = context_question
            t2t_label = answer

            fields = self.encode(
                tokenizer=tokenizer,
                input=input,
                label=t2t_label if self.text_to_text else None
            )

            if self.canonical:
                can_label = self.generate_span_labels(tokenizer, context_question, answer)
                if can_label == (-1, -1):
                    continue
                if can_label[0] >= self.max_seq_len - 1 or can_label[1] >= self.max_seq_len - 1:
                    continue
                fields['label'] = can_label

            features.append(fields)

        return self.convert_features_to_tensor_dataset(features, can_label_spans=self.canonical)


    def load_cnn_dailymail(self, tokenizer, mode):
        cnn_ds = self.load_summarization_features('cnn', tokenizer, mode)
        dm_ds = self.load_summarization_features('dailymail', tokenizer, mode)
        
        cnn_ds.extend(dm_ds)
        return self.convert_features_to_tensor_dataset(cnn_ds,
                                                       can_label_text=self.canonical)


    def load_summarization_features(self, dataset, tokenizer, mode):
        if mode == 'dev':
            mode = 'validation'
        if mode == 'train':
            mode = 'training'
        path = os.path.join(self.raw_data_dir, dataset, dataset, f'{mode}.jsonl')
       
        features = []

        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)

                input = line['context']
                input = self.prompt_template.encode_prompt('cnn_dailymail', input)
                label = line['answer']

                fields = self.encode(
                    tokenizer=tokenizer,
                    input=input,
                    label=label
                )

                features.append(fields)

        return features


    def load_multinli(self, tokenizer, mode):
        train_split = False
        if mode == 'train':
            train_split = True
        if mode == 'dev' or mode == 'train':
            mode = 'train'
            path = os.path.join(self.raw_data_dir, 'multinli',  'multinli_1.0', f'multinli_1.0_train.jsonl')
        elif mode == 'test':
            path = os.path.join(self.raw_data_dir, 'multinli', 'multinli_1.0', f'multinli_1.0_dev_matched.jsonl')

        all_nli = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)

                context = line['sentence1']
                question = line['sentence2']
                answer = line['gold_label']

                if '-' in answer:
                    continue

                all_nli.append((context, question, answer))

        if mode == 'train' and train_split:
            split = int(len(all_nli) * 0.95)
            all_nli = all_nli[:split]
        elif mode == 'train' and not train_split:
            split = int(len(all_nli) * 0.95)
            all_nli = all_nli[split:]

        features = []

        for nli in all_nli:
            context, hypothesis, answer = nli

            t2t_label = answer
            can_label = self.prompt_template.encode_label('multinli', t2t_label)

            input = self.prompt_template.encode_prompt('multinli', context, hypothesis)

            fields = self.encode(
                tokenizer=tokenizer,
                input=input,
                label=t2t_label if self.text_to_text else None
            )

            if self.canonical:
                fields['label'] = can_label

            features.append(fields)

        return self.convert_features_to_tensor_dataset(features,
                                                       can_label_class=self.canonical)
            

    def load_srl(self, tokenizer, mode):
        path = os.path.join(self.raw_data_dir, 'srl', f'{mode}.jsonl')

        features = []

        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)

                context = line['context']
                question = line['question']
                answer = line['answer']

                input = self.prompt_template.encode_prompt('srl', context, question)
                t2t_label = answer

                fields = self.encode(
                    tokenizer=tokenizer,
                    input=input,
                    label=t2t_label if self.text_to_text else None
                )

                if self.canonical:
                    # pulling out spans
                    can_label = self.generate_span_labels(tokenizer, context, answer)
                    if can_label == (-1, -1):
                        continue
                    if can_label[0] > self.max_seq_len or can_label[1] > self.max_seq_len:
                        continue
                    fields['label'] = can_label

                features.append(fields)

        return self.convert_features_to_tensor_dataset(features, can_label_spans=self.canonical)


    def load_zre(self, tokenizer, mode):
        path = os.path.join(self.raw_data_dir, 'zre', 'relation_splits', f'{mode}.jsonl')
        features = []

        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)

                context = line['context']
                question = line['question']
                answer = line['answer']


                input = self.prompt_template.encode_prompt('zre', context, question)
                t2t_label = answer

                fields = self.encode(
                    tokenizer=tokenizer,
                    input=input,
                    label=t2t_label if self.text_to_text else None
                )

                if self.canonical:
                    # pulling out spans

                    if answer == 'unanswerable':
                        can_label = (-1, -1)
                    else:
                        can_label = self.generate_span_labels(tokenizer, context, answer)
                        if can_label == (-1, -1):
                            continue
                        if can_label[0] > self.max_seq_len or can_label[1] > self.max_seq_len:
                            continue
                    fields['label'] = can_label
                
                features.append(fields)
    
        return self.convert_features_to_tensor_dataset(features, can_label_spans=self.canonical)


    def load_schema(self, tokenizer, mode):
        if mode == 'dev':
            mode = 'validation'
        path = os.path.join(self.raw_data_dir, 'schema', f'{mode}.jsonl')
        features = []

        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)

                context = line['context']
                question = line['question']
                answer = line['answer']

                input = self.prompt_template.encode_prompt('schema', context, question)
                t2t_label = answer

                fields = self.encode(
                    tokenizer=tokenizer,
                    input=input,
                    label=t2t_label if self.text_to_text else None
                )

                if self.canonical:
                    # pulling out spans

                    can_label = self.generate_span_labels(tokenizer, context, answer)
                    if can_label == (-1, -1):
                        continue
                    if can_label[0] > self.max_seq_len or can_label[1] > self.max_seq_len:
                        continue
                    fields['label'] = can_label
                
                features.append(fields)

        return self.convert_features_to_tensor_dataset(features, can_label_spans=self.canonical)


    def load_wikisql(self, tokenizer, mode):
        path = os.path.join(self.raw_data_dir, 'wikisql', 'data', f'{mode}.jsonl')
        table_path = os.path.join(self.raw_data_dir, 'wikisql', 'data', f'{mode}.tables.jsonl')

        with open(table_path) as tables_file:
            tables = [json.loads(line) for line in tables_file]
            id_to_tables = {x['id']: x for x in tables}

        all_answers = []
        examples = []
        count = 0
        features = []
        with open(path) as example_file:
            for idx, line in enumerate(example_file):
                entry = json.loads(line)
                human_query = entry['question']
                table = id_to_tables[entry['table_id']]
                sql = entry['sql']
                header = table['header']
                answer = repr(Query.from_dict(sql, header))
                input = (f'The table has columns {", ".join(table["header"])} ' +
                           f'and key words {", ".join(Query.agg_ops[1:] + Query.cond_ops + Query.syms)}')
                input += f'-- {human_query}'

                input = self.prompt_template.encode_prompt('wikisql', input)
                all_answers.append({'sql': sql, 'header': header, 'answer': answer, 'table': table})

                fields = self.encode(
                    tokenizer=tokenizer,
                    input=input,
                    label=answer
                )
                fields['wikisql_id'] = count

                features.append(fields)
                count += 1

        if mode == 'test':
            self.test_wikisql_answer = all_answers
        elif mode == 'train':
            self.train_wikisql_answer = all_answers
        elif mode == 'dev':
            self.dev_wikisql_answer = all_answers

        return self.convert_features_to_tensor_dataset(features,
                                                       can_label_text=self.canonical,
                                                       wiki_id=True)


    def convert_features_to_tensor_dataset(self, features,
                                           can_label_text=False,
                                           can_label_spans=False,
                                           can_label_class=False,
                                           wiki_id=False):
        # Converts a list of "fields" objects into tensors and a tensor dataset.
        # Note that different tasks have different features in the resulting dataset, which is mostly
        # handled by the `batch_to_inputs` function.
        # The 3 `can_label` arguments determine which type of canonical form to convert to.

        all_inputs = torch.tensor([f['inputs'] for f in features], dtype=torch.long)
        all_masks = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)

        if self.text_to_text or can_label_text or can_label_class:
            # If we are in a text-to-text setting, or the canonical label is text (seq2seq) or classification
            # we can handle it all the same.

            all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)

            if wiki_id:
                # wikisql is special

                wiki_ids = torch.tensor([f['wikisql_id'] for f in features], dtype=torch.long)
                return TensorDataset(all_inputs,
                                     all_masks,
                                     all_labels,
                                     wiki_ids)
            else:
                return TensorDataset(all_inputs,
                                     all_masks,
                                     all_labels)
        if can_label_spans:
            span_starts = torch.tensor([f['label'][0] for f in features], dtype=torch.long)
            span_ends = torch.tensor([f['label'][1] for f in features], dtype=torch.long)
            return TensorDataset(all_inputs,
                                 all_masks,
                                 span_starts,
                                 span_ends)

        raise Exception("Not sure how we got here!")



    def encode(self, *, tokenizer, input, label=None):
        # Generic tokenization function. Encodes inputs and labels, if passed.
        # Labels should only be passed if they are textual, e.g. if we are in text-to-text settings,
        # or if the task is seq2seq.

            input_encoded = tokenizer.encode_plus(
                input,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                return_attention_mask=True,
                pad_to_max_length=True
            )

            fields = {
                'inputs': input_encoded['input_ids'],
                'input_mask': input_encoded['attention_mask']
            }

            if label is not None:
                label_encoded = tokenizer.encode_plus(
                    label,
                    max_length=self.max_seq_len,
                    pad_to_max_length=True,
                    return_attention_mask=True
                )

                fields['label'] = label_encoded['input_ids']        

            return fields


    @classmethod
    def first_occurence_of(cls, seq, context):
        matches = {}
        # quick fix for some bug in tokenizer?
        seq = ['/' if t == ' /' else t for t in seq]
        seq = ['.' if t == ' .' else t for t in seq]
        if len(seq) == 0:
            return (-1, -1)

        for j, tok in enumerate(context):
    
            to_del = []
            for idx, i in matches.items():
                if seq[i] == tok:
                   matches[idx] += 1 
                else:
                    to_del.append(idx)
    
            for i in to_del:
                del matches[i]
    
            if tok == seq[0]:
                matches[j] = 1
    
            for idx, i in matches.items():
                if matches[idx] == len(seq):
                    return (idx, idx+matches[idx])
 
        return (-1, -1)


    @classmethod
    def generate_span_labels(cls, tokenizer, context, answer):
        tokenized_context = tokenizer.tokenize(context)
        tokenized_answer = tokenizer.tokenize(answer)

        return cls.first_occurence_of(
            tokenized_answer,
            tokenized_context
        )


    @classmethod
    def clean_punc(cls, seq):
        punc = '(){}-._,–"'
        return seq.translate(str.maketrans(punc, ' '*len(punc)))


    @classmethod
    def compute_metric_keys(cls, rouge=False, bleu=False, corpus_f1=False,
                            logical_form=False, dialogue=False):
        metric_keys = []
        if logical_form:
            metric_keys += ['lfem']
        if dialogue:
            metric_keys += ['joint_goal_em', 'turn_request_em', 'turn_goal_em', 'avg_dialogue']
        metric_keys += ['em']
        if bleu:
            metric_keys.append('bleu')
        if rouge:
            metric_keys += ['rouge1', 'rouge2', 'rougeL', 'avg_rouge']
        metric_keys.extend(['nf1', 'nem'])
        if corpus_f1:
            metric_keys += ['corpus_f1', 'precision', 'recall']
        return metric_keys


    @classmethod
    def task_metric_keys(cls, task):
        return cls.compute_metric_keys(
            bleu='iwslt' in task or 'multi30k' in task,
            dialogue='woz' in task,
            rouge='cnn' in task,
            logical_form='sql' in task,
            corpus_f1='zre' in task)


    @classmethod
    def _sample_train_batch(cls, dataset, batch_size):
        idcs = torch.randperm(len(dataset))[:batch_size]
        return dataset(idcs)

    @classmethod
    def clean_batch(cls, task, batch):
        if task == 'squad':
 
            span_start = batch[6]
            span_end = batch[7]

            mask = torch.ones(span_start.numel(), dtype=torch.bool)
            bad_idcs = []

            for i, ss in enumerate(span_start):
                se = span_end[i]
                if ss >= 255 or se >= 255:
                    print("hi")
                    bad_idcs.append(i)

            mask[bad_idcs] = False

            new_batch = [b[mask] for b in batch]
            return new_batch
        else:
            return batch



class Query:
    #https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10

    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']

    def __init__(self, sel_index, agg_index, columns, conditions=tuple()):
        self.sel_index = sel_index
        self.agg_index = agg_index
        self.columns = columns
        self.conditions = list(conditions)

    def __repr__(self):
        rep = 'SELECT {agg} {sel} FROM table'.format(
            agg=self.agg_ops[self.agg_index],
            sel= self.columns[self.sel_index] if self.columns is not None else 'col{}'.format(self.sel_index),
        )
        if self.conditions:
            rep +=  ' WHERE ' + ' AND '.join(['{} {} {}'.format(self.columns[i], self.cond_ops[o], v) for i, o, v in self.conditions])
        return ' '.join(rep.split())

    @classmethod
    def from_dict(cls, d, t):
        return cls(sel_index=d['sel'], agg_index=d['agg'], columns=t, conditions=d['conds'])
