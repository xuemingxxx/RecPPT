import copy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

class GPT2DataIterator(Dataset):
    def __init__(self, data: Dict, max_len: int, max_item: int, mask_prob: float, mask_id: int, pad_id: int,
                 device=torch.device('cpu'), partition = None):
        self.data = data
        self.max_len = max_len
        self.max_item = max_item
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.mask_prob = mask_prob
        self.device = device
        self.datasets = dict()
        self.dataset_partition = partition

    def __len__(self):
        return len(self.data)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def sample_function(self, user):
        sequence = self.data[user][0]
        if len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]
        seq = sequence[:-1]
        pos = sequence[1:]
        # seq_tensor = self._to_tensor(seq)
        # pos_tensor =self._to_tensor(pos)
        att = [0]*len(pos)
        return (seq, att, pos)

    def __getitem__(self, user: int) -> Tuple[Tensor, Tensor, Tensor]:
        sequence = copy.deepcopy(self.sample_function(user))
        datasets = copy.deepcopy(self.data[user][1])
        return (sequence[0], sequence[1], sequence[2], datasets)


class GPT2TestDataIterator(Dataset):
    def __init__(self, data: Dict, max_len: int, mask_id: int, pad_id: int,
                 device=torch.device('cpu')):
        self.data = data
        self.max_len = max_len
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.device = device
        self.index_dict = dict()
        key = 0
        for index in self.data.keys():
            self.index_dict[key] = index
            key = key + 1

    def __len__(self):
        return len(self.data)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def __getitem__(self, user: int) -> Tuple[Tensor, Tensor]:
        user = self.index_dict[user]
        sequence = self.data[user]['context']
        negative = self.data[user]['negative_sample']
        dataset = self.data[user]['DatasetName']

        if len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]
        att = [0]*len(sequence)
        # negative_tensor = self._to_tensor(negative_sample)
        return sequence, att, negative, dataset

def personl_define_collate_func(datas):
    def _to_tensor(value, dtype=torch.int64):
        return torch.tensor(value, dtype=dtype)
    seqs = [item[0] for item in datas]
    max_seq_length = max([len(seq) for seq in seqs])
    res = {'input_ids':[],'attention_ids':[], 'label_ids':[], 'datasets':[]}
    for item in datas:
        seq = item[0]
        attention_id = item[1]
        label_id = item[2]
        dataset_id = item[3]
        if len(seq) <= max_seq_length:
            res['input_ids'].append(seq + (max_seq_length - len(seq)) * [0])
            res['attention_ids'].append(attention_id + (max_seq_length - len(seq)) * [0])
            res['label_ids'].append(label_id + (max_seq_length - len(seq)) * [0])
        else:
            res['input_ids'].append(seq)
            res['attention_ids'].append(attention_id)
            res['label_ids'].append(label_id)
        res['datasets'].append(dataset_id)
    for key in res:
        data = res[key]
        res[key] = _to_tensor(data)
    return res['input_ids'], res['attention_ids'], res['label_ids'], res['datasets']

def personl_define_collate_func_test(datas):
    def _to_tensor(value, dtype=torch.int64):
        return torch.tensor(value, dtype=dtype)
    seqs = [item[0] for item in datas]
    max_seq_length = max([len(seq) for seq in seqs])
    res = {'input_ids':[],'attention_ids':[], 'label_ids':[], 'datasets':[]}
    for item in datas:
        seq = item[0]
        attention_id = item[1]
        label_id = item[2]
        dataset_id = item[3]
        if len(seq) <= max_seq_length:
            res['input_ids'].append(seq + (max_seq_length - len(seq)) * [0])
            res['attention_ids'].append(attention_id + (max_seq_length - len(seq)) * [0])
        else:
            res['input_ids'].append(seq)
            res['attention_ids'].append(attention_id)
        res['label_ids'].append(label_id)
        res['datasets'].append(dataset_id)
    for key in res:
        data = res[key]
        res[key] = _to_tensor(data)
    return res['input_ids'], res['attention_ids'], res['label_ids'], res['datasets']