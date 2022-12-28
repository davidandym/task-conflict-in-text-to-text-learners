# All Canonical Heads (and helper fns)



import copy

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_wrapper.functional import masked_mean_pooling
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import Seq2SeqLMOutput



class RegressionHead(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.linear = nn.Linear(config.d_model, 1)


    def forward(self, rep, labels=None):

        x = self.linear(rep)
        outputs = x
        if labels is not None:
            loss = F.mse_loss(x.squeeze(), labels)
            outputs = (loss, outputs)
        return outputs



class Classification2Head(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.linear = nn.Linear(config.d_model, 2)


    def forward(self, rep, labels=None):

        x = self.linear(rep)
        outputs = F.softmax(x, dim=1)
        if labels is not None:
            loss = F.nll_loss(F.log_softmax(x, dim=1), labels)
            outputs = (loss, outputs)
        return outputs



class Classification3Head(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.linear = nn.Linear(config.d_model, 3)


    def forward(self, rep, labels=None):

        x = self.linear(rep)
        outputs = F.softmax(x, dim=1)
        if labels is not None:
            loss = F.nll_loss(F.log_softmax(x, dim=1), labels)
            outputs = (loss, outputs)
        return outputs



def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    # Helper fn for Span Labeling.
    # Take the softmax of `logits` over given dimension, and set
    # entries to 0 wherever `mask` is 0.

    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs



class SpanLabelingHead(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.att_linear_1 = nn.Linear(config.d_model, 1)
        self.att_linear_2 = nn.Linear(config.d_model, 1)


    def forward(self,
                input_embeddings,
                input_mask,
                labels_span_start=None,
                labels_span_end=None):

        logits_1 = self.att_linear_1(input_embeddings)
        logits_2 = self.att_linear_2(input_embeddings)

        # Shapes: (batch_size, seq_len)
        pred1 = masked_softmax(logits_1.squeeze(), input_mask, log_softmax=True)
        pred2 = masked_softmax(logits_2.squeeze(), input_mask, log_softmax=True)

        criterion = nn.CrossEntropyLoss()

        loss = criterion(pred1, labels_span_start) + \
                criterion(pred2, labels_span_end)
        loss /= 2
        _, pred1_idx = pred1.max(dim=-1)
        _, pred2_idx = pred2.max(dim=-1)

        return loss, (pred1_idx, pred2_idx)



class NoAnswerSpanLabelingHead(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.att_linear_1 = nn.Linear(config.d_model, 1)
        self.att_linear_2 = nn.Linear(config.d_model, 1)
        self.no_answer_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.no_answer = nn.Linear(config.d_model, 2)


    def forward(self,
                input_embeddings,
                input_mask,
                labels_span_start=None,
                labels_span_end=None):

        logits_1 = self.att_linear_1(input_embeddings)
        logits_2 = self.att_linear_2(input_embeddings)
        sentence_embs = masked_mean_pooling(input_embeddings, input_mask, dim=1)

        # Shapes: (batch_size, seq_len)
        pred1 = masked_softmax(logits_1.squeeze(), input_mask, log_softmax=True)
        pred2 = masked_softmax(logits_2.squeeze(), input_mask, log_softmax=True)
        no_answer_logits = self.no_answer(sentence_embs)

        # true if "no answer" false if "answer"
        no_answer_label = labels_span_start == -1
        # 1 if "no answer" 0 if "answer"
        no_answer_label = no_answer_label.long()
        no_answer_loss = F.nll_loss(F.log_softmax(no_answer_logits, dim=1), no_answer_label)
        
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        loss = criterion(pred1, labels_span_start) + \
                criterion(pred2, labels_span_end)
        loss += no_answer_loss
        loss /= 3

        _, pred1_idx = pred1.max(dim=-1)
        _, pred2_idx = pred2.max(dim=-1)
        _, no_answer = no_answer_logits.max(dim=-1)

        return loss, (pred1_idx, pred2_idx, no_answer)



class SequenceOutputHead(nn.Module):
    # For autoregressive tasks.
    # The idea is to be a minimal LM Decoder Head with only the necessary helper functions and modules.
    # This head uses pre-trained parameters by first initializing a T5Stack, embedding,
    # and lm head with the proper config then copying the state of (already loaded) pre-trained parameters
    # for those modules.

    def __init__(self, config, pretrained_decoder, shared, lm_head):

        super().__init__()

        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.shared.load_state_dict(shared.state_dict())

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.load_state_dict(lm_head.state_dict())

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.decoder.load_state_dict(pretrained_decoder.state_dict())

       
    def _shift_right(self, input_ids):
        # Just a necessary function to copy over into this class.

        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


    def forward(
        self,
        encoder_outputs=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None
    ):

        if labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)
        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



# A const which defines the map between (benchmark, task) -> canonical head class
task_head_map = {
    'glue': {
        'cola': Classification2Head,
        'rte': Classification2Head,
        'mnli': Classification3Head,
        'mrpc': Classification2Head,
        'qnli': Classification2Head,
        'qqp': Classification2Head,
        'sst2': Classification2Head,
        'stsb': RegressionHead
    },
    'decanlp': {
        'multinli': Classification3Head,
        'sst': Classification2Head,
        'iwslt': SequenceOutputHead,
        'cnn_dailymail': SequenceOutputHead,
        'wikisql': SequenceOutputHead,
        'squad': SpanLabelingHead,
        'schema': SpanLabelingHead,
        'srl': SpanLabelingHead,
        'zre': NoAnswerSpanLabelingHead
    }
}
