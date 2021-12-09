import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers import AutoModel, BertModel, BertConfig
# from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from utils.layers import *
from utils.data_utils import get_gpt_token_num

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}

MODEL_CLASS_TO_NAME['albert'] =  list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}

#Add SapBERT configuration
model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_TO_CLASS[model_name] = 'bert'


class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states
        model_class = AutoModel
        self.module = model_class.from_pretrained(model_name, output_hidden_states=True)
        self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        input_ids, attention_mask, token_type_ids, output_mask = inputs
        outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]
        if self.output_token_states:
            return hidden_states, output_mask
        sent_vecs = hidden_states[:, 0]
        return sent_vecs, all_hidden_states
