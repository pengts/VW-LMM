from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..vw_pif_arch import vwMetaModel, vwPifMetaForCausalLM

class LlavaConfig(LlamaConfig):
    model_type = "llava"

class vwLlamaModel(vwMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(vwLlamaModel, self).__init__(config)


class vwPifLlamaForCausalLM(LlamaForCausalLM, vwPifMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = vwLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.vm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, image_position_labels, textual_image_labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        ###
        if image_position_labels is not None:
            image_position_labels=image_position_labels.bool()
            visual_labels=textual_image_labels
        else:
            visual_labels=None
        ###
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        loss_lm = None
        loss_vm = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct_lm = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_lm = loss_fct_lm(shift_logits, shift_labels)
            
        ###
        if visual_labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_visual_labels = visual_labels[..., 1:, :].contiguous()
            shift_image_position_labels = image_position_labels[..., 1:].contiguous()
            loss_fct_vm = KLDivLoss(reduction="batchmean")
            # loss_fct_vm = CrossEntropyLoss()
            shift_visual_logits = shift_logits[shift_image_position_labels]
            shift_visual_labels = shift_visual_labels[shift_image_position_labels]
            # shift_visual_logits = shift_logits
            assert shift_visual_logits.shape == shift_visual_labels.shape
            assert (shift_visual_labels>= 0).all()
            shift_visual_labels = shift_visual_labels.to(shift_visual_logits.device)
            
            # stage3
            # shift_visual_logits = nn.functional.softmax(shift_visual_logits,dim=-1)
            # shift_visual_labels = nn.functional.log_softmax(shift_visual_labels, dim=-1)
            # loss_vm = loss_fct_vm(shift_visual_labels, shift_visual_logits)

            # stage4
            shift_visual_logits = nn.functional.log_softmax(shift_visual_logits,dim=-1)
            # Since the labels used here are already softmaxed in prepare_inputs_labels_for_multimodal(), they are commented out here.
            # shift_visual_labels = nn.functional.softmax(shift_visual_labels, dim=-1)
            shift_visual_labels=shift_visual_labels.to(shift_visual_logits.dtype)
            loss_vm = loss_fct_vm(shift_visual_logits, shift_visual_labels)
            
            
        if loss_lm is not None and loss_vm is not None:
            loss = loss_lm + loss_vm
        elif loss_lm is None:
            loss = loss_vm
        elif loss_vm is None:
            loss = loss_lm
        else:
            loss = None
        ###

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

