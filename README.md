The official code repository for the EMNLP 2022 findings paper:

## ["Do Text-to-Text Learners Suffer From Multi-Task Conflict?"](https://aclanthology.org/2022.findings-emnlp.206/)


## Synopsis

The purpose of this paper was to explore how **reframing** multi-task learning (MTL) as a text-to-text problem can affect classic notions of transfer and task conflict. Traditionally, multi-task learning considers a set of tasks who share an input space (say, all images of size $128\times128$) but have (potentially) distinct output spaces. Thus, the **canonical** multi-task model considers a _shared encoder_ which encodes all inputs into a shared representation space, followed by a _task-specific decoder_ which uses the shared representation to make a task-specific prediction.
In this setting, we **specify** the task that we want to predict by selecting which task decoder will be used to make a prediction.

Recently in NLP, a different paradigm of multi-task learning has emerged: **text-to-text** multi-task learning considers a set of tasks who have a shared input space (all natural language sequences) and a _shared output space_ (all natural language sequences).
In this setting, the model considered is a _fully unified text-to-text_ model, with no task-specific parameters;
to specify which task we want the model to predict, we train the model to recognize **task prompts** which are prepended to task examples during training to ensure the model recognizes which tasks it is predicting into.

Text-to-text multi-task learners have exploded in popularity due to their impressive _zero-shot_ capabilities (their ability to generalize to unseen tasks and instructions), a form of generalization that does not typically arise in canonical multi-task learning.
Nevertheless, it may be desirable for our text-to-text models to be strong _multi-task generalizers_; that is, to generalize well to the tasks that they have been trained on.
Multi-task generalization has, traditionally, been a difficult problem: while training on diverse data may result in a better "meta-learner", this often comes at the cost of worse performance on the seen tasks, known as **negative transfer**.
Prior work has correlated negative transfer with **task conflict**, or significant differences between task gradients that prevents effective minimization of one or more task objectives.

Despite their popularity, the effect of text-to-text learning on negative transfer and task conflict has not been studied before.
If text-to-text learners _also_ suffer from task conflict and negative transfer, it follows that text-to-text learners may benefit from sophisticated multi-task optimization strategies.
To that end, we explore how different factors that emerge as we shift from canonical MTL to text-to-text MTL may affect multi-task transfer and task conflict.

Most of this code is a (hopefully) fairly straight-forward training harness for multi-task (and single-task) training (using T5 as the backbone for all models) across 2 English NLP benchmarks (DecaNLP and GLUE).

## Running Experiments

See `scripts/example.sh` for an example experiment script.
The entry point for any experiment script is `src/train.py`.
By default, models are not saved.
Instead, the results of each experiment are saved in two files: `{args.log_dir}/train_log.csv` and `{args.log_dir}/evaluation.json`.
`train_log.csv` will contain a csv of information from the entire training run. This is where you will find training loss curves, as well as measurements of task conflict.
`evaluation.json` will contain the final model performance on test and validation sets for all the tasks being considered.

To run the experiments from the paper, see `scripts/main_experiments/`.
In the paper, these experiments were repeated over multiple random seeds.

## Measuring Conflict

The code which measures conflict across tasks can be found in `src/conflict_measurements.py`. It operates in the following way:
1. A large batch gradient is computed for each task, by averaging the gradient of several smaller tasks.
   1. During this process, an approximation of the inter-task gradient covariance and noise is computed from the gradient norms of small and large batch gradients (see https://arxiv.org/abs/1812.06162).
2. The large batch gradient norm for each task is computed, and saved. Then the large batch gradients is normalized to have a norm of 1.
3. The normalized large batch gradients are averaged together. The resulting norm can be used to compute the average pairwise cosine similarity across all task gradients, which serves as our measurement of directional conflict.
4. The variance across the large batch gradients of each task is used to compute magnitude conflict.
5. Finally, the variance across inter-task gradient covariance is used to compute "noise conflict", a notion which may be expanded upon in future work (see https://openreview.net/forum?id=H9UOWMR_Ut).

The returned measurements of directional and magnitude conflict are used in the paper Figures 2, 3 and 4.

## Data

The data used for these experiments is open source, and should be freely available.
For GLUE experiments, I used the huggingface `datasets` package to download (and format) all data.
For DecaNLP experiments, I followed the downloading and formatting procedures put forth by the original DecaNLP repo (https://github.com/salesforce/decaNLP).

Datasets can be pre-loaded and cached with `src/data/cache_dataset.py`. For a given set of dataset-specifying arguments, this script will construct a dataset object and pre-process / tokenize all the data.
The resulting object, which is stored as a `.torch` object, can be loaded by setting `--precached_dataset_path` (in `train.py`) to point to it; doing so can save a decent chunk of time spent running the main experiment, especially if you are running multiple experiments over a single setting (dataset configuration).

## Evaluation

For GLUE tasks I use huggingface `load_metrics` and for DecaNLP I use the metrics proposed in the original , which can
be found in `deca_metrics.py`.

## Prompts

Part of this work studied how different prompts and output spaces may affect negative transfer and multi-task conflict.
In this project, variations in prompts and output spaces is only studied in GLUE.
The prompt and output space of GLUE can be set by the flags `--prompt_style` and `--label_style`.

GLUE has 3 available prompt styles, and 4 available output spaces.
The prompt styles are (1) _canonical_, which is the null prompt (no task specification); (2) _default_, which is a non-semantic task-specific token (such as "sst: {sentence}); (3) _multiprompt_, which are semantically rich, diverse prompts pulled from `promptsource` (such as "what is the sentiment of this sentence: {sentence}").

The output spaces available for GLUE (in the text-to-text setting) are: (1) _default_, which are the label tokens used in the original T5 paper; (2) _nonsemantic-overlap_, which consists of single-letter class tokens. The same 3 letters are used across all 8 classification tasks; (3) _nonsemantic-no-overlap_, which consists of single-letter class tokens. No letter is used more than once across all tasks; (4) _multiprompt_, which should be used when the inputs are also prompted with _multiprompt_ (because most prompts from promptsource specify the output space). For all settings, the outputs for `STS-B` are the same (`{:1.f}`).


For DecaNLP, controlling for prompt semantics and variation is more difficult because several of the tasks are framed as Q&A problems, in which the "task" to be performed is specified not by a predefined task prompt but by the _question_ portion of the input. This question is, by necessity, a semantically rich task prompt, and taking it out of the input would (ostensibly) make the task impossible.
A similar issue arises in the output space: only 2 tasks are classification tasks whose class labels may be set arbitrarily. All other tasks consist of output spaces who, in the text-to-text setting, are necessarily natural sequences of text.


## Citation

If you use this code, please cite:
```
@inproceedings{mueller-etal-2022-text,
    title = "Do Text-to-Text Multi-Task Learners Suffer from Task Conflict?",
    author = "Mueller, David  and
      Andrews, Nicholas  and
      Dredze, Mark",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.206",
    pages = "2843--2858",
    abstract = "Traditional multi-task learning architectures learn a single model across multiple tasks through a shared encoder followed by task-specific decoders. Learning these models often requires specialized training algorithms that address task-conflict in the shared parameter updates, which otherwise can lead to negative transfer. A new type of multi-task learning within NLP homogenizes multi-task architectures as a shared encoder and language model decoder, which does surprisingly well across a range of diverse tasks. Does this new architecture suffer from task-conflicts that require specialized training algorithms? We study how certain factors in the shift towards text-to-text models affects multi-task conflict and negative transfer, finding that both directional conflict and transfer are surprisingly constant across architectures.",
}
```
