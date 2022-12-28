The official code Repository for the EMNLP 2022 Findings Paper "Do Text-to-Text Learners Suffer From Multi-Task Conflict?"


## Synopsis

This project explores whether or not
reframing multi-task learning as a text-to-text problem
has a significant effect on observed negative transfer
and task conflict.

Multi-task learning is oftentimes synonymous with **negative transfer**, a term that describes when a system trained on multiple tasks underperforms a system trained on only a single task.
Negative transfer is often associated with **task conflict**: significant differences between task gradients (both in their directions and magnitudes), which leads optimization to favor certain tasks over others, and can result in.

The code to measure conflict can be found in
`src/conflict_measurements.py`.

Most of this code is a (hopefully) fairly straight-forward training harness for multi-task (and single-task) models across 2 standard benchmarks: DecaNLP and GLUE.

## Running Experiments

See `scripts/example.sh` for an example experiment script.
The entry point for any experiment script is `src/train.py`.
By default, models are not saved.
Instead, the results of each experiment are saved in two files: `{args.log_dir}/train_log.csv` and `{args.log_dir}/evaluation.json`.
`train_log.csv` will contain a csv of information from the entire training run. This is where you will find training loss curves, as well as measurements of task conflict.

To run the experiments from the paper, see `scripts/main_experiments/`.

The only other files that can be executed are the dataset files (`src/data/{benchmark}_dataset.py`).
The main function of these files will take a subset of the arguments from the main train script, construct the appropriate dataset, and then
cache it as a `.torch` object.
I would recommend doing this, generally, because parsing and tokenizing all of these tasks can take some time.
Moreover, you can reuse a multi-task dataset (a dataset constructed with `all-tasks`) for single-task experiments.

## Measuring Conflict

The code which measures conflict across tasks can be found in `src/conflict_measurements.py`. It operates in the following way:
1. A large batch gradient is computed for each task, by averaging the gradient of several smaller tasks.
   1. During this process, an approximation of the inter-task gradient covariance and noise is computed from the gradient norms of small and large batch gradients (see https://arxiv.org/abs/1812.06162).
2. The large batch gradient norm for each task is computed, and saved. Then the large batch gradients is normalized to have a norm of 1.
3. The normalized large batch gradients are averaged together. The resulting norm can be used to compute the average pairwise cosine similarity across all task gradients, which serves as our measurement of directional conflict.
4. The variance across the large batch gradients of each task is used to compute magnitude conflict.
5. Finally, the variance across inter-task gradient covariance is used to compute "noise conflict", a notion which may be expanded upon in future work (see https://openreview.net/forum?id=H9UOWMR_Ut).

The returned measurements of directional and magnitude conflict are used in the paper.

## Data

The data used for these experiments is open source, and should be freely available.
For GLUE experiments, I used the huggingface `datasets` package to download (and format) all data.
For DecaNLP experiments, I followed the downloading and formatting procedures put forth by the original DecaNLP repo (https://github.com/salesforce/decaNLP).

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

If you use this code, please cite: (The proper Bibtex should be forthcoming...)
```
@misc{https://doi.org/10.48550/arxiv.2212.06645,
  doi = {10.48550/ARXIV.2212.06645},
  
  url = {https://arxiv.org/abs/2212.06645},
  
  author = {Mueller, David and Andrews, Nicholas and Dredze, Mark},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Do Text-to-Text Multi-Task Learners Suffer from Task Conflict?},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```