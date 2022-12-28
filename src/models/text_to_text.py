# The Text-to-Text T5 Multi-Task model.



from transformers import T5ForConditionalGeneration



class Text2TextT5(T5ForConditionalGeneration):

    def forward(self, task, tokenizer,
                input_ids=None,
                attention_mask=None,
                labels=None,
                labels_mask=None,
                labels_span_start=None,
                labels_span_end=None,
                wiki_id=None):

        labels[labels == tokenizer.pad_token_id] = -100

        return super().forward(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)
