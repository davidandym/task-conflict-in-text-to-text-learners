# The Canonical T5 Multi-Task model.



from torch import nn
import torch.nn.functional as F
from pytorch_wrapper.functional import masked_mean_pooling
from transformers import T5ForConditionalGeneration

from canonical_heads import *



class CanonicalOutputs():
    # Class to mimic standard transformer output.

    def __init__(self, loss, outputs):

        self.loss = loss
        self.logits = outputs



class CanonicalEncoder(nn.Module):
    # Classic T5 Encoder Module.

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



class CanonicalT5(nn.Module):
    # The Canonical T5 Multi-Task Model Architecture.
    # The architecture consists of a shared pre-trained T5 encoder (& embeddings) and
    # unique, canonical heads for each task being considered.

    def __init__(self, args, config):

        super().__init__()

        self.config = config

        # Load a pre-trained T5 model (with decoder) for copying.
        pretrained_copy = T5ForConditionalGeneration.from_pretrained(
            args.pretrained_model,
            config=config
        )

        # Copy pre-trained embedding layer.
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.shared.load_state_dict(pretrained_copy.shared.state_dict())

        # Copy pre-trained encoder.
        self.encoder = CanonicalEncoder(args, config, pretrained_copy.encoder, self.shared)

        # Initialize Task-Specific Heads
        module_dict = {}
        for task in args.train_tasks:
            head_class = task_head_map[args.benchmark][task]
            if head_class == SequenceOutputHead:
                module_dict[task] = head_class(config,
                                               pretrained_copy.decoder,
                                               self.shared,
                                               pretrained_copy.lm_head)
            else:
                module_dict[task] = head_class(config)

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

        if type(self.task_specific_heads[task]) in [Classification2Head, Classification3Head, RegressionHead]:

            all_token_embs = encodings.last_hidden_state
            sentence_embs = masked_mean_pooling(all_token_embs, attention_mask, dim=1)
            outputs = self.task_specific_heads[task](sentence_embs, labels)
            return CanonicalOutputs(outputs[0], outputs[1])
        
        if type(self.task_specific_heads[task]) in [SpanLabelingHead, NoAnswerSpanLabelingHead]:

            embeds = encodings.last_hidden_state
            outputs = self.task_specific_heads[task](embeds,
                                                     attention_mask,
                                                     labels_span_start,
                                                     labels_span_end)
            return CanonicalOutputs(outputs[0], outputs[1])

        if type(self.task_specific_heads[task]) is SequenceOutputHead:

            hidden_states = encodings[0]
            labels[labels == tokenizer.pad_token_id] = -100
            return self.task_specific_heads[task](
                encoder_outputs=encodings,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                labels=labels
            )

        return None
