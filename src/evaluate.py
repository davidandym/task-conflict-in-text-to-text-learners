# Generic Validation Script



import torch
from datasets import load_metric

from utils import get_task_type
from deca_metrics import compute_metrics
from torch.utils.data import DataLoader, SequentialSampler



def evaluate(args, device, task, model, dataset, tokenizer, task_head=None, test=False):
    # The core function for evaluating a model on a given task and split.

    if task_head is None:
        task_head = task

    if test:
        val_set = dataset.dataset[task]['test']
    else:
        val_set = dataset.dataset[task]['dev']

    sampler = SequentialSampler(val_set)
    loader = DataLoader(val_set, sampler=sampler, batch_size=args.val_batch_size)

    all_predictions = []
    all_targets = []

    avg_total_loss = 0.
    num_steps = 0

    for batch in loader:

        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = dataset.batch_to_inputs(batch, task, args.canonical, args.prompted_inputs)
            outputs = model(task_head, tokenizer, **inputs)

        if args.canonical:
            predictions = get_canonical_predictions(args.benchmark, task, outputs.logits, inputs)
            targets = get_canonical_targets(args.benchmark, task, inputs, tokenizer)
        else:
            predictions = get_t2t_predictions(args.benchmark, task, outputs.logits, inputs)
            targets = get_t2t_targets(args.benchmark, task, inputs, tokenizer)

        avg_total_loss += float(outputs.loss)
        num_steps += 1

        all_predictions += predictions
        all_targets += targets


    if args.benchmark == 'glue':

        # For tasks with F1, we need to ensure that invalid predictions are explicitly incorrect,
        # not invalid. Huggingface throws an error otherwise. My simple solution is to just
        # make the prediction be one off the target label.
        if task == 'mrpc' or task == 'qqp':
            for i, l in enumerate(all_predictions):
                if l == -1:
                    all_predictions[i] = abs(all_targets[i] - 1)

        # Load huggingface metric for GLUE benchmark.
        metric = load_metric('glue', task, )
        metrics = metric.compute(predictions=all_predictions,
                                 references=all_targets)
    
    elif args.benchmark == 'decanlp':
        # For DecaNLP, I copied the evaluation metrics from the original MQAN model.
        # https://github.com/salesforce/decaNLP/blob/master/metrics.py

        metrics = compute_metrics(all_predictions, all_targets,
            bleu='iwslt' in task or 'multi30k' in task, dialogue='woz' in task,
            rouge='cnn' in task, logical_form='sql' in task, corpus_f1='zre' in task)
    else:
        raise Exception("Unknown Benchmark in Eval... how did you get here?")

    avg_total_loss /= num_steps
    metrics['loss'] = avg_total_loss
    return metrics



def get_t2t_predictions(benchmark, dataset, task, batch, outputs, tokenizer):
    
    task_type = get_task_type(benchmark, task)

    token_preds = torch.argmax(outputs.logits, dim=-1)
    predictions = tokenizer.batch_decode(token_preds)
    predictions = [sent.split('</s>')[0] for sent in predictions]

    if task_type.classification:

        decoded_predictions = []
        rev_idx = {v: k for k, v in dataset.prompt_template.label_templates.items()}
        
        for pred in predictions:
            if pred not in rev_idx:
                decoded_predictions.append(-1)
            else:
                decoded_predictions.append(rev_idx[pred])

        return decoded_predictions

    if task_type.regression:

        decoded_predictions = []
        for pred in predictions:
            try:
                decoded_predictions.append(float(pred))
            except ValueError:
                # I'm not certain that -1 is a great value, but it's what the original
                # T5 paper used for bad parses for STS-B, so I adopted it.
                # Certainly for any regression task with negative values this won't work.
                decoded_predictions.append(-1.)
        return decoded_predictions

    # If seq2seq or span-labeling task, just return the predicted string.
    return predictions



def get_canonical_predictions(benchmark, task, batch, outputs, tokenizer):
    
    task_type = get_task_type(benchmark, task)

    if task_type.classification:

        predictions = torch.argmax(outputs.logits, dim=-1).cpu()
        return predictions

    if task_type.regression:

        predictions = outputs.logits.squeeze()
        return predictions

    if task_type.seq2seq:

        token_preds = torch.argmax(outputs.logits, dim=-1).cpu()
        predictions = tokenizer.batch_decode(token_preds)
        predictions = [sent.split('</s>')[0] for sent in predictions]
        return predictions

    if task_type.span or task_type.span_noa:

        predictions = []

        span_start = outputs.logits[0]
        span_end = outputs.logits[1]

        if task_type.span_noa:
            no_answer = outputs.logits[2].cpu()

        inputs = batch['input_ids'].cpu()

        for i, input in enumerate(inputs):

            if task_type.span_noa:
                no_answer_pred = no_answer[i]
                if no_answer_pred == '1':
                    predictions.append("unanswerable")
                    continue

            ss, se = span_start[i], span_end[i]
            prediction = input[ss:se]
            prediction = tokenizer.decode(prediction)
            predictions.append(prediction)

        predictions = [sent.split('</s')[0] for sent in predictions]
        return predictions



def get_t2t_targets(benchmark, dataset, task, batch, tokenizer, mode):
    
    task_type = get_task_type(benchmark, task)

    if task == 'wikisql':
        # Special case WikiSQL, answer is pulled from dataset.

        targets = []
        for idx in batch['wiki_id'].cpu().int().tolist():
            if mode == 'dev':
                targets.append(dataset.dev_wikisql_answer[idx])
            elif mode == 'text':
                targets.append(dataset.text_wikisql_answer[idx])
        return targets

    targets = tokenizer.batch_decode(batch['labels'].cpu())
    targets = [sent.split('</s>')[0] for sent in targets]

    if task_type.classification:
        # If the task is classification we need to convert the decoded labels
        # into their canonical form.

        if task == 'mnli_mismatched':
            task = 'mnli'

        decoded_targets = []
        rev_idx = {v: k for k, v in dataset.prompt_template.label_templates.items()}
        
        for target in targets:
            decoded_targets.append(rev_idx[target])
        return decoded_targets

    if task_type.regression:
        # This target will be lower precision than the canonical model's target.
        # However, it represents the max precision that our T2T model is trained to predict.
        # Because this study is concerned with relative transfer, rather than comparing absolute
        # performance across T2T and Canonical settings, I feel this is reasonable.
        
        decoded_targets = [] 
        for target in targets:
            decoded_targets.append(float(target))
        return decoded_targets

    # If the task is not classification or regression, we can just return the decoded labels.
    return targets



def get_canonical_targets(benchmark, dataset, task, batch, tokenizer, mode):
    
    task_type = get_task_type(benchmark, task)

    if task == 'wikisql':
    # Special case WikiSQL, answer is pulled from dataset.

        targets = []
        for idx in batch['wiki_id'].cpu().int().tolist():
            if mode == 'dev':
                targets.append(dataset.dev_wikisql_answer[idx])
            elif mode == 'text':
                targets.append(dataset.text_wikisql_answer[idx])
        return targets

    if task_type.classification or task_type.regression:
        # For classification & regression, we can simply use the canonical labels
        # as is, converting them into a numpy array.

        labels = batch['labels'].cpu()
        return list(labels.cpu().numpy())

    if task_type.span:
        # Span labeling requires pulling out the gold span from
        # the input sequence.

        targets = []

        inputs = batch['input_ids'].cpu()
        gold_span_start = batch['labels_span_start'].cpu()
        gold_span_end = batch['labels_span_end'].cpu()

        for i, input in enumerate(inputs):
            ss, se = gold_span_start[i], gold_span_end[i]
            if ss == -1:
                target = "unanswerable"
            else:
                target = input[ss:se]
                target = tokenizer.decode(target)
            targets.append(target)

        targets = [sent.split('</s')[0] for sent in targets]
        return targets

    if task_type.seq2seq:

        labels = inputs['labels'].cpu()
        labels[labels == -100] = tokenizer.pad_token_id
        targets = tokenizer.batch_decode(labels)
        targets = [sent.split('</s>')[0] for sent in targets]
        return targets
