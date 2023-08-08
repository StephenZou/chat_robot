from typing import Optional, Union, Tuple

import torch
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss


class BertForNLU(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_intents = config.num_intents
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intents)
        self.domain_classifier = nn.Linear(config.hidden_size, config.num_domains)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slots)
        self.loss = CrossEntropyLoss()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        domain_logits = self.domain_classifier(pooled_output)
        # slot_logits = []
        # seq_len = last_hidden_state.size(1)
        tmp = []
        for x in attention_mask:
            x[0] = 0
            x[x.sum()-1] = 0
            tmp.append(x)
        tmp = torch.cat(tmp)
        mask = (tmp.view(1, -1) == 1).squeeze()
        slot_logits = last_hidden_state.view(-1, self.config.hidden_size)
        slot_logits = slot_logits[mask]
        slot_logits = self.dropout(slot_logits)
        slot_logits = self.slot_classifier(slot_logits)
        # for i in range(seq_len):
        #     slot_state = self.dropout(last_hidden_state[:, i, :].squeeze(1))
        #     slot_logits.append(self.slot_classifier(slot_state))
        # slot_logits = torch.cat(slot_logits)
        return intent_logits, domain_logits, slot_logits
