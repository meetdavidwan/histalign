import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2PreTrainedModel, GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, NLLLoss

from typing import Optional, Tuple, Union, Any

import numpy as np

from torch_scatter.composite import scatter_logsumexp


LARGE_NEG = -10000.

"""
Code modifed from Huggingface's GPT2LMHeadModel code
and adapted from Trime: https://github.com/princeton-nlp/TRIME/blob/main/fairseq/criterions/trime_loss.py
"""
class GPT2LMHeadModelWithCache(GPT2LMHeadModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # From this point new code
        # create shifted labels for cache, uses input_ids for generation
        cache_labels = torch.roll(input_ids, -1, dims=-1)
        cache_labels[:,-1] = 0 # This ensures that we do not shift pad token to the last and mask out the whole word

        similarity = self.calc_similarity(hidden_states)

        # create mask and mask out tokens we cannot attend to
        bz, sz = hidden_states.size(0), hidden_states.size(1)
        local_mask = torch.triu(torch.ones((bz,sz,sz), device=similarity.device), diagonal=0).bool()
        if self.config.pad_token_id is not None:
            # pad_mask = input_ids == self.config.pad_token_id
            pad_mask = cache_labels == self.config.pad_token_id
            pad_mask = pad_mask.unsqueeze(-1) | pad_mask.unsqueeze(1)
            local_mask = local_mask | pad_mask
        similarity[local_mask] = LARGE_NEG

        # need to manually reset the mask part again esepcially for fp16
        # otherwise the score will not be small enough for scatter logsumexp and pad token will get very large prob
        similarity_norm = F.log_softmax(similarity, dim=-1).clone()
        similarity_norm[local_mask] = LARGE_NEG

        lm_logits = self.get_logits_from_hidden(hidden_states, similarity)
        
        loss = None
        if labels is not None:
            loss = self.calc_loss(lm_logits, similarity, similarity_norm, input_ids, labels, local_mask)
        else:
            lm_logits = self.get_logits(lm_logits, similarity, similarity_norm, input_ids)
        
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
    def calc_similarity(self, hidden_states):
        hidden_states_scaled = hidden_states / float(hidden_states.size(-1)) ** 0.25
        similarity = torch.bmm(hidden_states_scaled, hidden_states_scaled.transpose(-1,-2))
        return similarity
    
    def calc_loss(self, lm_logits, similarity, similarity_norm, labels, local_mask):
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def get_logits(self, lm_logits, similarity, similarity_norm, input_ids):
        return lm_logits
    
    def get_logits_from_hidden(self, hidden_states, similarity):
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


class GPT2LMHeadModelWithTrime(GPT2LMHeadModelWithCache):
    def calc_loss(self, lm_logits, similarity, similarity_norm, input_ids, labels, local_mask):
        # token loss
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )

        # context loss based on labels
        negatives = shift_labels.unsqueeze(-1) != shift_labels.unsqueeze(1)
        ctx_loss = - torch.logsumexp(similarity_norm[:,:-1,:-1] + negatives * LARGE_NEG, dim=-1)

        # normalize
        norm_c = torch.logsumexp(similarity[:,:-1,:-1], dim=-1)
        norm_t = torch.logsumexp(shift_logits, dim=-1)
        norm_tpc = torch.logsumexp(torch.stack((norm_t, norm_c), dim=-1), dim=-1)
        
        # calculate normalized loss for each token
        ctx_loss, norm_c, norm_t, norm_tpc = ctx_loss.view(-1), norm_c.view(-1), norm_t.view(-1), norm_tpc.view(-1)
        loss = -torch.logsumexp(torch.stack((-token_loss + norm_t - norm_tpc, -ctx_loss + norm_c - norm_tpc), dim=-1), dim=-1)

        # Take the average across all tokens
        labels_mask = (shift_labels != -100).view(-1)
        loss = loss[labels_mask].sum() / labels_mask.sum()
        return loss

    def get_logits(self, lm_logits, similarity, similarity_norm, input_ids):
        norm_c = torch.logsumexp(similarity, dim=-1)
        norm_t = torch.logsumexp(lm_logits, dim=-1)
        norm_tpc = torch.logsumexp(torch.stack((norm_t, norm_c), dim=-1), dim=-1)

        lm_logits = F.log_softmax(lm_logits, dim=-1)
        
        cache_labels = torch.roll(input_ids, -1, dims=-1)
        cache_labels[:,-1] = 0 # This ensures that we do not shift pad token to the last and mask out the whole word

        index = cache_labels.unsqueeze(1).expand(-1,cache_labels.size(1),-1)

        cache_logits = torch.full(lm_logits.shape, LARGE_NEG, dtype=similarity.dtype, device=similarity.device)
        cache_logits = scatter_logsumexp(src=similarity_norm, dim=-1, index=index, out=cache_logits)
        cache_logits = cache_logits.nan_to_num(LARGE_NEG)

        norm_c, norm_t, norm_tpc = norm_c.unsqueeze(-1), norm_t.unsqueeze(-1), norm_tpc.unsqueeze(-1)
        comb_logits = torch.logsumexp(torch.stack((lm_logits + norm_t - norm_tpc, cache_logits + norm_c - norm_tpc), dim=-1), dim=-1)

        # lm_logits = lm_logits
        lm_logits = cache_logits
        # lm_logits = comb_logits

        return lm_logits

class GPT2LMHeadModelWithHistALign(GPT2LMHeadModelWithTrime):

    def calc_loss(self, lm_logits, similarity, similarity_norm, input_ids, labels, local_mask):
        trime_loss = super().calc_loss(lm_logits, similarity, similarity_norm, input_ids, labels, local_mask)

        similarity = similarity[:,:-1,:-1]
        local_mask = local_mask[:,:-1,:-1]

        shift_labels = labels[..., 1:].contiguous()

        shift_cache_labels = input_ids[..., 1:].contiguous() # similar to shift labels but no -100
        negatives = shift_cache_labels.unsqueeze(-1) != shift_cache_labels.unsqueeze(1)

        # calculate soft labels with embedding
        emb_labels = self.transformer.wte(shift_cache_labels)
        emb_weight = self.compute_emb_weight(emb_labels)

        # we only want to calculate contrastive loss for instances
        # that has positive examples (has cache labels)
        labels_mask = (shift_labels != -100)
        has_cache = ((~negatives & ~local_mask).sum(-1) > 0)

        comb_mask = (~local_mask) & labels_mask.unsqueeze(-1)
        comb_mask = comb_mask & has_cache.unsqueeze(-1)


        emb_weight_indices = torch.argsort(emb_weight, descending=True, dim=-1)
        
        similarity_sorted = torch.gather(similarity, dim=-1, index=emb_weight_indices)
        emb_weight_sorted = torch.gather(emb_weight, dim=-1, index=emb_weight_indices)
        mask_sorted = torch.gather(comb_mask, dim=-1, index=emb_weight_indices)

        neg_mask_sorted = torch.gather(negatives, dim=-1, index=emb_weight_indices)

        ones = torch.ones_like(similarity_sorted, device=similarity_sorted.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        # ranking_loss = loss_func(similarity_sorted, similarity_sorted, ones)
        ranking_loss = []

        # excluding true positives and limit to a rank
        # n = min(similarity_sorted.size(-1), self.config.max_margin_rank) if self.config.max_margin_rank is not None else similarity_sorted.size(-1)
        n = similarity_sorted.size(-1)
        for i in range(1, n):
            max_pos = -i
            max_neg = n

            pos_score = similarity_sorted[..., :max_pos].contiguous().view(-1)
            neg_score = similarity_sorted[..., i:max_neg].contiguous().view(-1)

            ones = torch.ones_like(pos_score, device=similarity_sorted.device)

            loss_func = torch.nn.MarginRankingLoss(self.config.margin * i, reduction="none")
            loss = loss_func(pos_score, neg_score, ones)

            # padding mask
            pos_mask = mask_sorted[..., :max_pos].contiguous().view(-1)
            neg_mask = mask_sorted[..., i:max_neg].contiguous().view(-1)
            loss_mask = torch.logical_and(pos_mask, neg_mask)

            # positive values should not be ranked
            pos_mask = neg_mask_sorted[...,:max_pos].contiguous().view(-1)
            neg_mask = neg_mask_sorted[...,i:max_neg].contiguous().view(-1)

            comb_mask = ~pos_mask & neg_mask # only predict when its a positive negative pair
            loss_mask = loss_mask & comb_mask

            loss = loss[loss_mask]

            if loss.numel() > 0:
                # ranking_loss += loss.mean()
                ranking_loss.append(loss.mean())
                
        ranking_loss = torch.mean(torch.stack(ranking_loss))
        loss = trime_loss + self.config.contrastive_weight * ranking_loss

        return loss

    def compute_emb_weight(self, emb_labels):
        # emb_labels = emb_labels / float(emb_labels.size(-1)) ** 0.25
        emb_labels = F.normalize(emb_labels, dim=-1)

        emb_weight = torch.bmm(emb_labels, emb_labels.transpose(-1,-2))
        # emb_weight = F.log_softmax(emb_weight, dim=-1)
        return emb_weight