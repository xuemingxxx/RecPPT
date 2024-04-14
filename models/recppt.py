import math
import numpy as np
from typing import List, Optional, Tuple, Union
import random

import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class ReprogramEncoder(nn.Module):
    def __init__(self, config, D=768):
        super().__init__()
        assert config.hidden_units%config.numHeader  == 0
        self.v_size = 300
        self.nhead = config.numHeader
        self.dim = config.hidden_units
        self.D = D
        # self.seq_len = config.seq_len
        self.dim_ = self.dim//self.nhead
        self.proto_linear_k = nn.Linear(self.D, self.dim, bias=False)
        self.proto_linear_v = nn.Linear(self.D, self.dim, bias=False)
        self.item_linear_q = nn.Linear(self.dim, self.dim, bias=False)
        self.prototypes_variable = nn.Embedding(self.v_size, self.D, sparse=False)
        self.dropout = torch.nn.Dropout(p=config.dropout_rate)

    def init_emb(self):
        initrange = 0.5 / self.dim
        self.prototypes_variable.weight.data.uniform_(-initrange, initrange)
    
    def init_emb_with_gpt2(self, wte):
        gpt2_v_size = wte.shape[0]
        index = random.sample([i for i in range(0, gpt2_v_size)], self.v_size)
        self.prototypes_variable = torch.nn.Embedding.from_pretrained(wte[index].clone())

    def forward(self, item_embedding):
        batch_size, seq_len, dim = item_embedding.shape
        proto = torch.tile(torch.Tensor([range(self.v_size)]).int().to(item_embedding.device), (batch_size, 1))
        proto_embs = self.prototypes_variable(proto) # [B, v_size, dim]
        # proto_embs = self.dropout(proto_embs)
        mha_k = self.proto_linear_v(proto_embs).reshape((-1, self.nhead, self.v_size, self.dim_)) # [B, nhead, v_size, dim_]
        mha_q = self.item_linear_q(item_embedding).reshape((-1, self.nhead, seq_len ,self.dim_)) # [B, nhead, seq_len, dim_]
        mha_v = self.proto_linear_k(proto_embs).reshape((-1, self.nhead, self.dim_, self.v_size)) # [B, nhead, dim_, v_size]
        scores = torch.softmax(torch.matmul(mha_q, mha_k.transpose(2, 3))/math.sqrt(self.dim_), dim=-1) # [B, nhead, seq_len, v_size]
        z = torch.reshape(torch.matmul(scores, mha_v.transpose(2, 3)), (batch_size, seq_len, self.nhead*self.dim_)) # [B, nhead, dim_, seq_len] -> # [B, seq_len, dim]
        return z


class RecPPT(nn.Module):
    def __init__(self, user_num, item_num, args, device):
        super(RecPPT, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.device = device
        self.is_gpt = True
        self.freeze = False
        self.action = args.action
        if self.action == 'fewshot':
            self.reprogramming = ReprogramEncoder(args)
        if self.is_gpt:
            self.gpt2 = GPT2Model.from_pretrained('./model/gpt2/', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            self.gpt2.h = self.gpt2.h[:args.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))

        self.gpt2_hidden_units = 768
        self.item_emb = torch.nn.Embedding(self.item_num, args.hidden_units, padding_idx=0).requires_grad_(True)
        self.linear = torch.nn.Linear(self.gpt2_hidden_units, self.item_num, bias=False)

        if self.freeze:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln_' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        self.dropout = nn.Dropout(args.dropout_rate)

    def feats(self, x):
        attention_mask = x.clone()
        attention_mask[attention_mask!=0] = 1
        outputs = self.item_emb(x)
        if self.action == 'fewshot':
            outputs = self.reprogramming(outputs)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs, attention_mask=attention_mask).last_hidden_state
        outputs = self.linear(outputs)
        return outputs

    def forward(self, x):
        feats = self.feats(x)
        outputs = feats
        return outputs

    def predict(self, x):
        attention_mask = x.clone()
        attention_mask[attention_mask!=0] = 1
        out = self.forward(x)
        dummie = torch.zeros((out.size(0), 1), device=self.device)
        attention_mask = torch.concat((attention_mask, dummie), dim=1)
        indices = torch.argmin(attention_mask, dim=1, keepdim=False) - 1
        return out[torch.arange(out.size(0), device=self.device), indices]
