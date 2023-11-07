from typing import Optional, Union, Tuple

import torch
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from CRF import CRF


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
        self.crf = CRF(config.slot2index)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            slots: Optional[torch.Tensor] = None,
            intents: Optional[torch.Tensor] = None,
            domains: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        intent_logits, domain_logits, slot_logits = self._compute_logit(input_ids, attention_mask, token_type_ids,
                                                                        position_ids, head_mask, inputs_embeds,
                                                                        output_attentions, output_hidden_states,
                                                                        return_dict)
        # 将[CLS]和[SEP]遮盖掉
        for x in attention_mask:
            x[x.sum() - 1] = 0
            x[0] = 0
        slot_loss = self.crf.neg_log_likelihood(slot_logits, slots, attention_mask)
        batch_loss = (self.criterion(intent_logits, intents) + self.criterion(domain_logits, domains) + slot_loss) / 3
        batch_intent_acc = (torch.argmax(intent_logits, dim=1) == intents).float().mean()
        batch_domain_acc = (torch.argmax(domain_logits, dim=1) == domains).float().mean()
        _, best_slots = self.crf(slot_logits, attention_mask)
        # slots = (slots*attention_mask+(attention_mask-1))[:, :-1]
        best_slots = best_slots*attention_mask[:, :-1] + (attention_mask[:, :-1]-1)
        batch_slot_acc = torch.tensor([all(x) for x in best_slots[:, 1:] == slots]).float().mean()
        return batch_loss, batch_intent_acc, batch_domain_acc, batch_slot_acc

    def _compute_logit(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
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
            return_dict=return_dict
        )
        last_hidden_state = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        domain_logits = self.domain_classifier(pooled_output)
        slot_logits = self.dropout(last_hidden_state)
        slot_logits = self.slot_classifier(slot_logits)
        return intent_logits, domain_logits, slot_logits

    def predict(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None
    ) -> Tuple[torch.Tensor]:
        intent_logits, domain_logits, slot_logits = self._compute_logit(input_ids, attention_mask,
                                                                        return_dict=return_dict)
        _, best_slots = self.crf(slot_logits)
        return torch.argmax(intent_logits), torch.argmax(domain_logits), best_slots
