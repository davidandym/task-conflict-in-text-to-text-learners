# Code to measure conflict in the directions and magnitudes of gradients across tasks.
# As a word of warning, interpreting these notions of conflict (computed at the gradient level)
# feels akin to reading tea-leaves.
# While they presented an interesting analysis in our work, I'd be careful taking too much out of
# these quantities alone.



from collections import defaultdict

import numpy as np
import torch



def get_conflict_metric_keys(train_tasks, args):
    # Grab all keys which can be logged, so the CSV logger can write them in the header.

    keys = []
    for task in train_tasks:
        keys += [
            f'{task} gradient norm',
            f'{task} covariance trace',
            f'{task} noise scale',
        ]

    keys += [
        'directional conflict',
        'magnitude conflict',
        'covariance conflict',
        'noise conflict'
    ]

    if args.measure_conflict_decoder:
        for task in train_tasks:
            keys += [
                f'{task} decoder gradient norm',
                f'{task} decoder covariance trace',
                f'{task} decoder noise scale',
            ]

        keys += [
            'decoder directional conflict',
            'decoder magnitude conflict',
            'decoder covariance conflict',
            'decoder noise conflict'
        ]

    return keys



def measure_conflict(args,
                     train_tasks,
                     model,
                     tokenizer,
                     opt,
                     device,
                     dataset,
                     batch_big=1024,
                     batch_small=16,
                     measure_decoder=False):
    # At a high-level, this function aims to quantify the magnitude of difference between different task gradients
    # during multi-task training.
    # There are 3 quantities by which this is measured:
    #   1. Direction (avg. pairwise cosine similarity)
    #   2. Magnitude (The covariance of gradient magnitudes across tasks)
    #   3. Noise (The covariance of inter-task noise levels across tasks; noise level here means the inter-task gradient covariance)
    # By default these quantities are often measured only over the shared encoder, since that component is constant across canonical and T2T models.
    # However, by setting measure_decoder to True, the function will separately record conflict in the decoder for T2T models.

    # This will store all of the results.
    metrics = {}

    # These quantities are responsible for holding the aggregated multi-task gradient as we move throught the tasks.
    averaged_encoder_gradient = None
    averaged_decoder_gradient = None

    for task in train_tasks:
        # For each task we compute a "big batch" gradient, which is computed by averaged the gradient of several smaller batchs together.
        # This accomplishes a couple things:
        #   1. By considering a larger batch over all tasks, we have a more accurate representation of the gradient w.r.t. direction.
        #   2. We get an estimate of the inter-task gradient noise "for free".
        #   3. It allows us to reduce the variance of the individual task gradient norms by computing them over several small batches.
        #       Unfortunately, we didn't to this last one for the paper, and instead used the gradient norm of the big-batch, which doesn't
        #       represent the norm of the task gradient used for training. Nevertheless, this is a potential benefit. I've left the code here
        #       to represent the settings of the original experiment, but it would be easy to compute magnitude conflict across the average small-batch
        #       gradient norms rather than the big-batch gradient norm.
        # Before being returned, the big batch gradient is normalized to norm 1 for directional conflict calculation.

        gradient, covariance_trace, noise_scale, gradient_norm, \
            decoder_gradient, decoder_covariance_trace, decoder_noise_scale, decoder_gradient_norm = compute_big_batch_gradient(
            args,
            task,
            dataset,
            model,
            tokenizer,
            opt,
            device,
            batch_big,
            batch_small,
            collect_decoder=measure_decoder
        )

        metrics[f'{task} covariance trace'] = covariance_trace
        metrics[f'{task} gradient norm'] = gradient_norm
        metrics[f'{task} noise scale'] = noise_scale
        if averaged_encoder_gradient is None:
            averaged_encoder_gradient = gradient
        else:
            averaged_encoder_gradient.add_(gradient)

        if measure_decoder:

            metrics[f'{task} decoder covariance trace'] = decoder_covariance_trace
            metrics[f'{task} decoder gradient norm'] = decoder_gradient_norm
            metrics[f'{task} decoder noise scale'] = decoder_noise_scale
            if averaged_decoder_gradient is None:
                averaged_decoder_gradient = decoder_gradient
            else:
                averaged_decoder_gradient.add_(decoder_gradient)


    # After we are done computing the big-batch gradient for each task, and have the aggregated task gradient,
    # we can compute our 3 conflict metrics.

    # We average the 1-norm task gradients together.
    averaged_encoder_gradient.mul_(float(1)/len(train_tasks))
    # The resulting gradient norm is equal to (up to a constant) the average pairwise cosine similarity across all task pairs.
    directional_conflict = averaged_encoder_gradient.norm()
    # Magnitude conflcit is the variance of gradient norms across tasks.
    magnitude_conflict = np.var([metrics[f'{t} gradient norm'] for t in train_tasks])
    # Finally, the covariance (noise) conflict are the variance of the task covariance trace (noise levels) across tasks.
    # (noise level is gradient covariance trace normalized by expected gradient norm: see appendix of https://arxiv.org/abs/1812.06162)
    covariance_conflict = np.var([metrics[f'{t} covariance trace'] for t in train_tasks])
    noise_conflict = np.var([metrics[f'{t} noise scale'] for t in train_tasks])

    # Add all metrics to the log.
    metrics['directional conflict'] = float(directional_conflict)
    metrics['magnitude conflict'] = magnitude_conflict
    metrics['covariance conflict'] = covariance_conflict
    metrics['noise conflict'] = noise_conflict

    if measure_decoder:
        averaged_decoder_gradient.mul_(float(1)/len(train_tasks))
        decoder_directional_conflict = averaged_decoder_gradient.norm()
        decoder_magnitude_conflict = np.var([metrics[f'{t} decoder gradient norm'] for t in train_tasks])
        decoder_covariance_conflict = np.var([metrics[f'{t} decoder covariance trace'] for t in train_tasks])
        decoder_noise_conflict = np.var([metrics[f'{t} decoder noise scale'] for t in train_tasks])

        metrics['decoder directional conflict'] = float(decoder_directional_conflict)
        metrics['decoder magnitude conflict'] = decoder_magnitude_conflict
        metrics['decoder covariance conflict'] = decoder_covariance_conflict
        metrics['decoder noise conflict'] = decoder_noise_conflict

    return metrics



def compute_big_batch_gradient(args,
                               task,
                               dataset,
                               model,
                               tokenizer,
                               opt,
                               device,
                               big_batch_size=1024,
                               small_batch_size=16,
                               collect_decoder=False):
    # This function computes a large-batch gradient for a given task by averaging it over many smaller batch gradients.
    # In addition, the function also computes approximate covariance statistics for the task, and normalizes the large-batch
    # gradient before returning.

    train_set = dataset.dataset[task]['train']
    dataset_size = len(train_set)

    big_batch_size =  min(dataset_size, big_batch_size)
    big_batch_size = (big_batch_size // small_batch_size) * small_batch_size

    idcs = torch.randperm(dataset_size)[:big_batch_size]
    big_batch = train_set[idcs]

    big_batch_grad = None
    small_batch_norms = []

    decoder_big_batch_grad = None
    decoder_small_batch_norms = []

    num_splits = int(big_batch_size / small_batch_size)

    # set model into eval mode -- turns off dropout, which may affect conflict.
    model.eval()

    for i in range(num_splits):
        opt.zero_grad()

        batch = tuple([b[i*small_batch_size:i*small_batch_size+small_batch_size] for b in big_batch])
        batch = tuple(t.to(device) for t in batch)
        inputs = dataset.batch_to_inputs(batch, task)
        out = model(task, tokenizer, **inputs)
                    # input_ids=inputs['input_ids'],
                    # attention_mask=inputs['attention_mask'],
                    # labels=inputs['labels'])
        loss = out.loss
        loss.backward()

        gradient = collect_gradients(model, args.measure_conflict_on_cpu)
        small_batch_norms.append(float(gradient.norm()))

        if big_batch_grad is None:
            big_batch_grad = gradient
        else:
            big_batch_grad.add_(gradient)

        if collect_decoder:
            decoder_gradient = collect_gradients(model, args.measure_conflict_on_cpu, decoder=True)
            decoder_small_batch_norms.append(float(gradient.norm()))

            if decoder_big_batch_grad is None:
                decoder_big_batch_grad = gradient
            else:
                decoder_big_batch_grad.add_(gradient)


    opt.zero_grad()
    big_batch_grad.mul_(1/float(num_splits))

    big_batch_norm = np.square(float(big_batch_grad.norm()))
    small_batch_norm = np.mean(np.square(small_batch_norms))
    approximate_covariance_trace = (1 / ((1 / small_batch_size) - (1 / big_batch_size))) * (small_batch_norm - big_batch_norm)
    expected_gradient_norm = (1 / (big_batch_size - small_batch_size)) * ((big_batch_size * big_batch_norm) - (small_batch_size * small_batch_norm))
    noise_scale = approximate_covariance_trace / expected_gradient_norm

    big_batch_grad.mul_(1/big_batch_grad.norm())

    if collect_decoder:

        decoder_big_batch_grad.mul_(1/float(num_splits))

        decoder_big_batch_norm = np.square(float(decoder_big_batch_grad.norm()))
        decoder_small_batch_norm = np.mean(np.square(decoder_small_batch_norms))
        decoder_approximate_covariance_trace = (1 / ((1 / small_batch_size) - (1 / big_batch_size))) * (decoder_small_batch_norm - decoder_big_batch_norm)
        decoder_expected_gradient_norm = (1 / (big_batch_size - small_batch_size)) * ((big_batch_size * decoder_big_batch_norm) - (small_batch_size * decoder_small_batch_norm))
        decoder_noise_scale = decoder_approximate_covariance_trace / decoder_expected_gradient_norm

        decoder_big_batch_grad.mul_(1/big_batch_grad.norm())

        return (big_batch_grad,
                approximate_covariance_trace,
                noise_scale,
                big_batch_norm,
                decoder_big_batch_grad,
                decoder_approximate_covariance_trace,
                decoder_noise_scale,
                decoder_big_batch_norm)
    else:
        return (big_batch_grad,
                approximate_covariance_trace,
                noise_scale,
                big_batch_norm,
                None,
                None,
                None,
                None) # stubs to match the signature when decoder is included.



def collect_gradients(model, on_cpu=True, decoder=False):
    
    gradients = []

    for n, p in model.named_parameters():
        if should_collect_gradient(n, p, decoder):
            if on_cpu:
                gradients.append(p.grad.flatten().cpu())
            else:
                gradients.append(p.grad.flatten())

    flattened = torch.cat(gradients, dim=-1)
    return flattened



def should_collect_gradient(parameter_name, parameter, include_layer_norm=False, decoder=False):

    if not parameter.requires_grad or parameter.grad is None:
        return False

    if decoder:
        if "decoder" in parameter_name:
            if not include_layer_norm and 'layer_norm' in parameter_name:
                return False
            print(parameter_name)
            return True

    else:
        if "encoder" in parameter_name:
            if not include_layer_norm and 'layer_norm' in parameter_name:
                return False
            return True
