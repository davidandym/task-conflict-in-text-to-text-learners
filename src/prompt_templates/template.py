# Prompt & Label Template Class



class PromptTemplate():


    prompt_templates = {}
    label_templates = {}

    
    def __init__(self, args):

        self.prompt = args.prompt_style
        self.labels = args.label_style
        
        self.prompt_templates = self.load_prompts(args)
        self.label_templates = self.load_labels(args)


    def load_prompts(self):
        # Override in subclass, to determine which templates to load.

        pass


    def load_labels(self):
        # Override in subclass, to determine which label templates to load.

        pass


    def encode_prompt(self, task, s1, s2=None):

        if s2 is None:
            return self.prompt_templates.format(s1)
        else:
            return self.prompt_templates.format(s1, s2)


    def encode_label(self, task, label):

        return self.label_templates[task][label]
