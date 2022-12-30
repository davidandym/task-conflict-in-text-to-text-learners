# Templates, pre-fixes, and answer words for GLUE tasks used in T5



import random

from promptsource.templates import DatasetTemplates

from prompt_templates.template import PromptTemplate



canonical_templates = {
    'cola': '{}',
    'rte':  '{} {}',
    'mnli': '{} {}',
    'mrpc': '{} {}',
    'qnli': '{} {}',
    'qqp':  '{} {}',
    'sst2':  '{}',
    'stsb': '{} {}'
}


default_templates = {
    'cola': 'cola sentence: {}',
    'rte':  'rte sentence1: {}   sentence2: {}',
    'mnli': 'mnli hypothesis: {}   premise: {}',
    'mrpc': 'mrpc sentence1: {}   sentence2: {}',
    'qnli': 'qnli question: {}   sentence: {}',
    'qqp':  'qqp question1: {}   question2: {}',
    'sst2':  'sst2 sentence: {}',
    'stsb': 'stsb sentence1: {}   sentence2: {}'
}


multiprompt_templates = {
    'cola': 1,
    'rte':  1,
    'mnli': 2,
    'mrpc': 6,
    'qnli': 1,
    'qqp':  2,
    'sst2':  0,
    'stsb': 0
}


default_answer_maps = {
    'cola': {
        0: 'unacceptable',
        1: 'acceptable'
    },
    'rte': {
        0: 'entailment',
        1: 'not_entailment'
    },
    'mnli': {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    },
    'mrpc': {
        0: 'not_equivalent',
        1: 'equivalent'
    },
    'qnli': {
        0: 'entailment',
        1: 'not_entailment'
    },
    'qqp': {
        0: 'not_duplicate',
        1: 'duplicate'
    },
    'sst2': {
        0: 'negative',
        1: 'positive'
    },
    'stsb': '{:.1f}'
}


multiprompt_answer_maps = {
    'cola': {
        0: 'no',
        1: 'yes'
    },
    'rte': {
        0: 'yes',
        1: 'no'
    },
    'mnli': {
        0: 'yes',
        1: 'maybe',
        2: 'no'
    },
    'mrpc': {
        0: 'no',
        1: 'yes'
    },
    'qnli': {
        0: 'yes',
        1: 'no'
    },
    'qqp': {
        0: 'no',
        1: 'yes'
    },
    'sst2': {
        0: 'negative',
        1: 'positive'
    },
    'stsb': '{:.1f}'
}


nonsemantic_no_overlap_answer_maps = {
    'cola': {
        0: 'z',
        1: 'y'
    },
    'rte': {
        0: 'w',
        1: 'v'
    },
    'mnli': {
        0: 'u',
        1: 't',
        2: 's'
    },
    'mrpc': {
        0: 'r',
        1: 'q'
    },
    'qnli': {
        0: 'p',
        1: 'o'
    },
    'qqp': {
        0: 'n',
        1: 'm'
    },
    'sst2': {
        0: 'l',
        1: 'k'
    },
    'stsb': '{:.1f}'
}


nonsemantic_overlap_answer_maps = {
    'cola': {
        0: 'z',
        1: 'y'
    },
    'rte': {
        0: 'z',
        1: 'y'
    },
    'mnli': {
        0: 'z',
        1: 'x',
        2: 'y'
    },
    'mrpc': {
        0: 'z',
        1: 'y'
    },
    'qnli': {
        0: 'z',
        1: 'y'
    },
    'qqp': {
        0: 'z',
        1: 'y'
    },
    'sst2': {
        0: 'z',
        1: 'y'
    },
    'stsb': '{:.1f}'
}



class GlueTemplate(PromptTemplate):
    
    def __init__(self, args):

        super().__init__(args)

    
    def load_prompts(self):

        if self.prompt == "canonical":
            return canonical_templates
        if self.prompt == "default":
            return default_templates
        if self.prompt == "multiprompt":
            return multiprompt_templates
        
        raise Exception(f"Unknown prompt style: {self.prompt}")

    
    def load_labels(self):

        if self.labels == "default":
            return default_answer_maps
        if self.labels == "multiprompt":
            return multiprompt_answer_maps
        if self.labels == "nonsemantic-overlap":
            return nonsemantic_overlap_answer_maps
        if self.labels == "nonsemantic-no-overlap":
            return nonsemantic_no_overlap_answer_maps

        raise Exception(f"Unknown label style: {self.labels}")


    def encode_prompt(self, task, s1, s2=None, example=None, mode='train'):

        if self.prompt == "multiprompt":
            # If multi-prompt training, then some special cases,
            # e.g. fixed test & dev cases.

            if task == 'mnli_mismatched':
                templates = DatasetTemplates("glue", 'mnli')
            else:
                templates = DatasetTemplates("glue", task)

            template_list = [v for _, v in templates.templates.items()]

            if mode == 'train':
                template = random.choice(template_list)
            else:
                template = template_list[multiprompt_templates[task]]

            result = template.apply(example)
            if len(result) == 1:
                return None
            else:
                return result[0]

        return super().encode_prompt(task, s1, s2)


    def encode_label(self, task, label, example=None, mode='train'):

        if self.labels in ["default", "nonsemantic-overlap", "nonsemantic-no-overlap"]:
            if task == 'stsb':
                return self.label_templates[task].format(label)
            else:
                return self.label_templates[task][label]

        elif self.labels == "multiprompt":
            # If multi-prompt training, then some special cases,
            # e.g. fixed test & dev cases.

            if task == 'mnli_mismatched':
                templates = DatasetTemplates("glue", 'mnli')
            else:
                templates = DatasetTemplates("glue", task)

            template_list = [v for _, v in templates.templates.items()]

            if mode == 'train':
                template = random.choice(template_list)
            else:
                template = template_list[multiprompt_templates[task]]

            result = template.apply(example)
            if len(result) == 1:
                return None
            else:
                return result[1]
