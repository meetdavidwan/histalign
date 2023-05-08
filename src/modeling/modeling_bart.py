from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

import numpy as np

from transformers.models.bart.modeling_bart import shift_tokens_right, BartModel, BartDecoder, BartDecoderLayer, BartAttention, BartConfig

from torch_scatter.composite import scatter_logsumexp

LARGE_NEG = -10000.


class BartForConditionalGenerationWithCache(BartForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_input_ids = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids if encoder_outputs is None else None, # we still want to keep input_ids
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.encoder_last_hidden_state
        decoder_hidden_states = outputs.last_hidden_state
        similarity = self.calc_similarity(encoder_hidden_states, decoder_hidden_states)

        # Create mask and mask out tokens we cannot attend to.
        # Since we are looking at encoder, all tokens are vaild. Only pad
        local_mask = torch.zeros(
            (
                encoder_hidden_states.size(0),
                decoder_hidden_states.size(1),
                encoder_hidden_states.size(1)
            ),
            dtype=torch.bool,
            device=encoder_hidden_states.device
        )

        pad_mask = (input_ids if input_ids is not None else cache_input_ids) == self.config.pad_token_id
        pad_mask = pad_mask.unsqueeze(1).expand(-1, decoder_hidden_states.size(1) ,-1)

        # for generation, we may need to extend for beam
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, -1, local_mask.size(0)//pad_mask.size(0))
        pad_mask = pad_mask.reshape(-1, pad_mask.size(1), pad_mask.size(2))

        local_mask[pad_mask] = True

        similarity[local_mask] = LARGE_NEG

        similarity_norm = F.log_softmax(similarity, dim=-1).clone()
        similarity_norm[local_mask] = LARGE_NEG

        lm_logits = self.get_logits_from_hidden(decoder_hidden_states, encoder_hidden_states, similarity)

        loss = None
        if labels is not None:
            loss = self.calc_loss(lm_logits, similarity, similarity_norm, input_ids, decoder_input_ids, labels, local_mask)
        else:
            lm_logits = self.get_logits(lm_logits, similarity, similarity_norm, cache_input_ids)
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def calc_similarity(self, encoder_hidden_states, decoder_hidden_states):
        encoder_hidden_scaled = encoder_hidden_states / float(encoder_hidden_states.size(-1)) ** 0.25
        decoder_hidden_scaled = decoder_hidden_states / float(decoder_hidden_states.size(-1)) ** 0.25
        similarity = torch.bmm(decoder_hidden_scaled, encoder_hidden_scaled.transpose(-1,-2))
        return similarity
    
    def calc_loss(self, lm_logits, similarity, similarity_norm, input_ids, decoder_input_ids, labels, local_mask):
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss
    
    def get_logits_from_hidden(self, decoder_hidden_states, encoder_hidden_states, similarity):
        lm_logits = self.lm_head(decoder_hidden_states)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        return lm_logits

    def get_logits(self, lm_logits, similarity, similarity_norm, input_ids):
        return lm_logits
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "cache_input_ids": kwargs["cache_input_ids"],
        }
    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        # we still want input_ids
        model_kwargs["cache_input_ids"] = inputs_tensor
        return model_kwargs

class BartForConditionalGenerationWithTrime(BartForConditionalGenerationWithCache):
    def calc_loss(self, lm_logits, similarity, similarity_norm, input_ids, decoder_input_ids, labels, local_mask):
        # token loss
        token_loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1),
            reduction='none'
        )

        # context loss based on labels
        negatives = labels.unsqueeze(-1) != input_ids.unsqueeze(1)
        ctx_loss = - torch.logsumexp(similarity_norm + negatives * LARGE_NEG, dim=-1)

        # normalize
        norm_c = torch.logsumexp(similarity, dim=-1)
        norm_t = torch.logsumexp(lm_logits, dim=-1)
        norm_tpc = torch.logsumexp(torch.stack((norm_t, norm_c), dim=-1), dim=-1)
        
        # calculate normalized loss for each token
        ctx_loss, norm_c, norm_t, norm_tpc = ctx_loss.view(-1), norm_c.view(-1), norm_t.view(-1), norm_tpc.view(-1)
        loss = -torch.logsumexp(torch.stack((-token_loss + norm_t - norm_tpc, -ctx_loss + norm_c - norm_tpc), dim=-1), dim=-1)

        # Take the average across all tokens
        labels_mask = (labels != -100).view(-1)
        loss = loss[labels_mask].sum() / labels_mask.sum()
        return loss

    def get_logits(self, lm_logits, similarity, similarity_norm, input_ids):
        norm_c = torch.logsumexp(similarity, dim=-1)
        norm_t = torch.logsumexp(lm_logits, dim=-1)
        norm_tpc = torch.logsumexp(torch.stack((norm_t, norm_c), dim=-1), dim=-1)

        lm_logits = F.log_softmax(lm_logits, dim=-1)
        
        index = input_ids.unsqueeze(1).expand(-1,similarity.size(1),-1)
        index = index.unsqueeze(-1).expand(-1, -1, -1, similarity.size(0)//input_ids.size(0))
        index = index.reshape(-1, index.size(1), index.size(2))

        cache_logits = torch.full(lm_logits.shape, LARGE_NEG, dtype=similarity.dtype, device=similarity.device)
        cache_logits = scatter_logsumexp(src=similarity_norm, dim=-1, index=index, out=cache_logits)
        cache_logits = cache_logits.nan_to_num(LARGE_NEG)

        norm_c, norm_t, norm_tpc = norm_c.unsqueeze(-1), norm_t.unsqueeze(-1), norm_tpc.unsqueeze(-1)
        comb_logits = torch.logsumexp(torch.stack((lm_logits + norm_t - norm_tpc, cache_logits + norm_c - norm_tpc), dim=-1), dim=-1)

        # lm_logits = lm_logits
        # lm_logits = cache_logits
        lm_logits = comb_logits

        return lm_logits

class BartForConditionalGenerationWithHistAlign(BartForConditionalGenerationWithTrime):
    def calc_loss(self, lm_logits, similarity, similarity_norm, input_ids, decoder_input_ids, labels, local_mask):
        trime_loss = super().calc_loss(lm_logits, similarity, similarity_norm, input_ids, decoder_input_ids, labels, local_mask)

        negatives = labels.unsqueeze(-1) != input_ids.unsqueeze(1)

        # calculate soft labels with embedding
        emb_inputs = input_ids
        emb_labels = torch.roll(decoder_input_ids, -1, dims=-1) # prevent labels with -100

        emb_inputs = self.model.shared(emb_inputs)
        emb_labels = self.model.shared(emb_labels)

        emb_weight = self.compute_emb_weight(emb_inputs, emb_labels, negatives).detach()

        emb_weight_indices = torch.argsort(emb_weight, descending=True, dim=-1)
        
        similarity_sorted = torch.gather(similarity, dim=-1, index=emb_weight_indices)
        emb_weight_sorted = torch.gather(emb_weight, dim=-1, index=emb_weight_indices)

        # create mask we dont want paddings
        labels_mask = (labels != -100)
        has_cache = ((~negatives & ~local_mask).sum(-1) > 0)

        comb_mask = (~local_mask) & labels_mask.unsqueeze(-1)
        comb_mask = comb_mask & has_cache.unsqueeze(-1)

        mask_sorted = torch.gather(comb_mask, dim=-1, index=emb_weight_indices)
        neg_mask_sorted = torch.gather(negatives, dim=-1, index=emb_weight_indices)

        ones = torch.ones_like(similarity_sorted, device=similarity_sorted.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        ranking_loss = loss_func(similarity_sorted, similarity_sorted, ones)

        for i in range(1, similarity_sorted.size(-1)):

            pos_score = similarity_sorted[..., :-i].contiguous().view(-1)
            neg_score = similarity_sorted[..., i:].contiguous().view(-1)

            ones = torch.ones_like(pos_score, device=similarity_sorted.device)

            loss_func = torch.nn.MarginRankingLoss(self.config.margin * i, reduction="none")
            loss = loss_func(pos_score, neg_score, ones)

            # padding mask
            pos_mask = mask_sorted[..., :-i].contiguous().view(-1)
            neg_mask = mask_sorted[..., i:].contiguous().view(-1)
            loss_mask = torch.logical_and(pos_mask, neg_mask)
            
            pos_mask = neg_mask_sorted[...,:-i].contiguous().view(-1)
            neg_mask = neg_mask_sorted[...,i:].contiguous().view(-1)

            comb_mask = ~pos_mask & neg_mask

            loss_mask = loss_mask & comb_mask

            loss = loss[loss_mask].mean()
            ranking_loss += loss

        loss = trime_loss + self.config.contrastive_weight * ranking_loss 

        return loss

    def compute_emb_weight(self, emb_inputs, emb_labels, negatvies):
        # emb_inputs = emb_inputs / float(emb_inputs.size(-1)) ** 0.25
        # emb_labels = emb_labels / float(emb_labels.size(-1)) ** 0.25

        emb_inputs = F.normalize(emb_inputs, dim=-1)
        emb_labels = F.normalize(emb_labels, dim=-1)

        emb_weight = torch.bmm(emb_labels, emb_inputs.transpose(-1,-2))
        # emb_weight = F.log_softmax(emb_weight, dim=-1)
        return emb_weight