# Prompt and label tempaltes for the DecaNLP benchmark.
# Unlike GLUE, we only have default prompts and labels for this set.



from prompt_templates.template import PromptTemplate



canonical_templates = {
    "iwslt":            "{}",
    "cnn_dailymail":    "{}",
    "wikisql":          "{}",
    "multinli":         "{} {}",
    "sst":              "{}",
    "srl":              "{} {}",
    "zre":              "{} {}",
    "schema":           "{} {}",
    "squad":            "{} {}",
}


# Note that a lot of prompts are simply both {} {}. This is because
# For the tasks that are already Q\&A tasks, the task to be done is
# (i.e. the prompt) is specified by the "question" part of the input
# already.
default_templates = {
    "iwslt":            "{} Translate from English to German",
    "cnn_dailymail":    "{} What is the summary?",
    "wikisql":          "{} What is the translation from English to SQL?",
    "multinli":         "Context: {} Premise: {} -- entailment, neutral, or contradiction?",
    "sst":              "{} Is this review negative or positive?",
    "srl":              "{} {}",
    "zre":              "{} {}",
    "schema":           "{} {}",
    "squad":            "{} {}",
}


# Only the 2 classification tasks have intentional label maps.
# All other tasks have natural textual outputs.
default_label_templates = {
    "multinli":         {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    },
    "sst":              {
        0: 'negative',
        1: 'positive'
    },
}



class DecaNLPTemplate(PromptTemplate):

    def __init__(self, args):

        super().__init__(args)


    def load_prompts(self):

        if self.prompt == 'canonical':
            return canonical_templates
        if self.prompt == 'default':
            return default_templates

        raise Exception(f"Unkown prompt style: {self.prompt}")


    def load_labels(self):
        return default_label_templates
