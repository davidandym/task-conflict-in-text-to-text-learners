# The Text-to-Text T5 Multi-Task model with Independent Task Heads (T2T-ID).



import copy

from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack

from canonical_heads import SequenceOutputHead



class CanonicalEncoder(nn.Module):

    def __init__(self, config, pretr_enc, pretr_shared):
        super().__init__()

        self.config = config

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, pretr_shared)
        self.encoder.load_state_dict(pretr_enc.state_dict())

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs



class Text2TextIndependentHeadsT5(nn.Module):
    # A Text-to-Text model with independent decoder heads for each task.

    def __init__(self, args, config):

        super().__init__()

        pretrained_copy = T5ForConditionalGeneration.from_pretrained(
            args.pretrained_model,
            config=config
        )

        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.shared.load_state_dict(pretrained_copy.shared.state_dict())

        self.encoder = CanonicalEncoder(config, pretrained_copy.encoder, self.shared)

        module_dict = {}
        for task in args.train_tasks:
            module_dict[task] = SequenceOutputHead(config,
                                                   pretrained_copy.decoder,
                                                   self.shared,
                                                   pretrained_copy.lm_head)
        
        self.task_specific_heads = nn.ModuleDict(module_dict)


    def forward(self, task, tokenizer,
                input_ids=None,
                attention_mask=None,
                labels=None,
                labels_mask=None,
                labels_span_start=None,
                labels_span_end=None,
                wiki_id=None):

        encodings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = encodings[0]
        labels[labels == tokenizer.pad_token_id] = -100

        return self.task_specific_heads[task](
            encoder_outputs=encodings,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            labels=labels
        )
