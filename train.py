import argparse
import os
import sys
from typing import  Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adagrad, Adadelta, Adam, AdamW
from torch.nn.functional import softmax
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from transformers import (
    get_scheduler
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from model.metrics import RecallAtK, nDCG
from model.recppt import RecPPT
from common.utils import DotDict
from parser import args
from utils.ds_utils import get_train_ds_config
from utils.utils import get_optimizer_grouped_parameters
from utils.dataset_utils import personl_define_collate_func, personl_define_collate_func_test
from utils.dataset_utils import GPT2DataIterator, GPT2TestDataIterator

seed = 60
torch.manual_seed(seed)
np.random.seed(seed)
bce_criterion = torch.nn.BCEWithLogitsLoss()
pad_token_id = 0
loss_func_reduce = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id) 
loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none') 
    
def transfer_model_dict(model_state_dict):
    state_dict = dict()
    for key in list(model_state_dict.keys()):
        state_dict[key.replace('module.', '')] = model_state_dict.pop(key)
    return state_dict

def filter_model_dict(model_state_dict):
    state_dict = dict()
    for key in list(model_state_dict.keys()):
        if 'item_emb' in key or 'wte' in key:
        # if 'item_emb' in key in key:
            model_state_dict.pop(key)
            continue
        state_dict[key] = model_state_dict.pop(key)
        #state_dict[key.replace('module.', '')] = model_state_dict.pop(key)
    return state_dict

def aggregate_items(df, pad_id, length, dataset_dict):
    """ aggregate user's items """
    df.sort_values(['user_id', 'Timestamp'], inplace=True)
    df = df.groupby('user_id').apply(lambda x: [x['item_id'].tolist(), dataset_dict[x['DatasetName'].head(1).to_string().split()[1]]])
    for idx in df.index:
        df[idx][0] = df[idx][0][-length:]
    return df

def init_model(model, model_state_dict):
    state_dict = dict()
    for key in list(model_state_dict.keys()):
        if 'embeddings' in key or 'cls' in key or 'pooler' in key:
            continue
        state_dict[key.replace('beta', 'weight').replace('gamma', 'bias')] = model_state_dict.pop(key)
    return state_dict

def get_softmax_embedding(dataset_partitons, num_item):
    dataset_count = len(dataset_partitons.keys())
    biggest_number = 65504.0
    softmax_mask = torch.full((dataset_count, num_item), biggest_number) 
    for i in dataset_partitons.keys():
        min_id, max_id = dataset_partitons[i]
        for id in range(min_id, max_id+1):
            softmax_mask[i, id] = 0
    softmax_mask_emb = torch.nn.Embedding.from_pretrained(softmax_mask)
    return softmax_mask_emb

def train(model, epoch, train_dataloader, valid_dataloaders, test_dataloaders, optim, k=[1], metrics=[], callback=[], params=None, init_item_emb=None, device=None, checkpoint_path=None, softmax_embedding=None):
    best_metric = {name:0 for name in test_dataloaders.keys()}
    current_metric = 0
    print(model)
    for e in range(epoch):
        # ------ train --------
        train_loss = 0
        total_step = len(train_dataloader)
        history = {}

        for step, (seq, attention, label, dataset_ids) in enumerate(train_dataloader):
            if step%params.num_gpus != params.global_rank:
                continue
            if params.action == 'fewshot' and step > (params.fewshotSize/params.batchSize - 1):
                continue
            seq, attention, label, dataset_ids = seq.to(device), attention.to(device), label.to(device), dataset_ids.to(device)
            
            if params.model == 'recppt':
                seq, _, pos = seq, attention, label 
                logits = model.forward(seq)
                mask = softmax_embedding(dataset_ids).unsqueeze(1)
                mask = mask.expand(mask.shape[0], seq.shape[1], mask.shape[2])
                logits_mask = logits-mask
                logits_mask = logits_mask.reshape(-1, logits_mask.size(-1))
                pos = pos.reshape(-1)
                losses = loss_func(logits_mask, pos)
                ratios = []
                dataset_ids = dataset_ids + params.dataset_count
                with torch.no_grad():
                    for i in range(params.dataset_count, params.dataset_count*2):
                        ratio = dataset_ids.clone().unsqueeze(1).float()
                        ratio[ratio!=i] = 0
                        ratio[ratio==i] = 1
                        ratio = ratio.expand(dataset_ids.shape[0], seq.shape[1]).flatten()
                        ratios.append(ratio)
                loss_tasks = [losses*ratio for ratio in ratios]
                loss_list = [sum(l)/torch.count_nonzero(l) for l in loss_tasks if torch.count_nonzero(l)!=0 ]
                line = 'single_loss:' + ' '.join([str(l.item()) for l in loss_list])
                print(line)
                loss = torch.pow(torch.prod(torch.stack(loss_list)), 1/len(loss_list))

            # print('learning_rate:', optim.get_lr())
            model.backward(loss)
            model.step()
            train_loss += loss.item()

            # ------ step end ------
            history['epoch'] = e + 1
            history['train_loss'] = loss
            print(f"Epoch {e} Step {step}/{total_step} training_loss : {loss.item():3.3f}")

            # ------ valid --------
            if params.global_rank == 0 and (step % params.val_step ==0 or step == total_step):
                model.eval()
                y_pred, y_true = {i:[] for i in k}, {i:[] for i in k}
                with torch.no_grad():
                    for name in valid_dataloaders.keys():
                        val_loss = 0
                        y_pred, y_true = {i:[] for i in k}, {i:[] for i in k}
                        for step_test, (seq, attention, negative, dataset_ids) in enumerate(valid_dataloaders[name]):
                            seq, attention, negative, dataset_ids = seq.to(device), attention.to(device), negative.to(device), dataset_ids.to(device)
                            label = negative[:,0]
                            loss = torch.Tensor([0.0])
                            # random shuffle
                            idx = torch.randperm(negative.shape[1])
                            negative = negative[:,idx]

                            if params.model == 'recppt':
                                preds = model.predict(seq)
                                preds = torch.gather(preds, dim=1, index=negative)

                            val_loss += loss.item()
                            for i in k:
                                # top k index
                                _, indices = torch.topk(preds, k=i)
                                # indexing samples to get top k sample
                                pred = torch.gather(negative, dim=1, index=indices)
                                y_pred[i].extend(pred.cpu().tolist())
                                y_true[i].extend(label.cpu().tolist())

                        history['valid_loss'] = val_loss / step_test
                        result = f"Epoch {e} Step {step} {name} valid_loss : {history['valid_loss']:4.4f} "

                        for func in metrics:
                            for i in k:
                                metrics_value = func(y_pred[i], y_true[i])
                                history[f'{func}'] = metrics_value
                                result += f' valid_{func}@{i} : {metrics_value:4.4f}'
                                if type(func) is nDCG:
                                    current_metric = metrics_value
                        print(result, flush=True)

                        if best_metric[name] < current_metric and params.action!='fewshot':
                            best_metric[name] = current_metric
                            model_name = 'best_acc_model_layer{}_head{}_dim{}_seed{}_{}.pth'.format(params.numLayer,params.numHeader,params.dimPerHeader,seed,name)
                            torch.save(model.state_dict(), os.path.join(checkpoint_path, model_name))
                            print("Update Best Model at Epoch {} Step {} Name {} for dataset {}".format(e, step, model_name, name), flush=True)
            
            # ------ test --------
            if params.global_rank == 0 and (step % params.val_step ==0 or step == total_step):
                model.eval()
                y_pred, y_true = {i:[] for i in k}, {i:[] for i in k}
                with torch.no_grad():
                    for name in test_dataloaders.keys():
                        val_loss = 0
                        y_pred, y_true = {i:[] for i in k}, {i:[] for i in k}
                        for step_test, (seq, attention, negative, dataset_ids) in enumerate(test_dataloaders[name]):
                            seq, attention, negative, dataset_ids = seq.to(device), attention.to(device), negative.to(device), dataset_ids.to(device)
                            label = negative[:,0]
                            loss = torch.Tensor([0.0])
                            # random shuffle
                            idx = torch.randperm(negative.shape[1])
                            negative = negative[:,idx]

                            if params.model == 'recppt':
                                preds = model.predict(seq)
                                preds = torch.gather(preds, dim=1, index=negative)

                            val_loss += loss.item()
                            for i in k:
                                # top k index
                                _, indices = torch.topk(preds, k=i)
                                # indexing samples to get top k sample
                                pred = torch.gather(negative, dim=1, index=indices)
                                y_pred[i].extend(pred.cpu().tolist())
                                y_true[i].extend(label.cpu().tolist())

                        history['test_loss'] = val_loss / step_test
                        result = f"Epoch {e} Step {step} {name} test_loss : {history['test_loss']:4.4f} "

                        for func in metrics:
                            for i in k:
                                metrics_value = func(y_pred[i], y_true[i])
                                history[f'{func}'] = metrics_value
                                result += f' test_{func}@{i} : {metrics_value:4.4f}'
                                if type(func) is nDCG:
                                    current_metric = metrics_value
                        print(result, flush=True)          
            model.train()

            if step >= total_step:
                break


if __name__ == '__main__':
    argument = args()
    print(argument, flush=True)

    if argument.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(argument.local_rank)
        device = torch.device("cuda", argument.local_rank)
        deepspeed.init_distributed()
    global_rank = torch.distributed.get_rank()
    print('global_rank', global_rank)
    num_gpus = torch.cuda.device_count()
    print('num_gpus', num_gpus)
    argument.offload = True
    ds_config = get_train_ds_config(offload=argument.offload,
                                    stage=argument.zero_stage)
    ds_config['train_micro_batch_size_per_gpu'] = argument.per_device_train_batch_size
    ds_config['train_batch_size'] = argument.per_device_train_batch_size * torch.distributed.get_world_size() * argument.gradient_accumulation_steps
    torch.distributed.barrier()

    datasets = argument.dataset.split(',')
    dataset_dict = dict(zip(datasets, range(len(datasets))))
    datasets_name = '_'.join(datasets)
    data_dir = './datasets/processed/'
    train_data = pd.read_csv(os.path.join(data_dir, datasets_name + '_' + 'train.tsv'), sep='\t', low_memory=False)
    valid_data = pd.read_csv(os.path.join(data_dir, datasets_name + '_' + 'valid.tsv'), sep='\t', low_memory=False)
    test_data = pd.read_csv(os.path.join(data_dir, datasets_name + '_' + 'test.tsv'), sep='\t', low_memory=False)

    # pad_token_id = min(min(train_data['item_id']), min(valid_data['item_id']), min(test_data['item_id'])) - 1
    pad_token_id = 0
    mask_token_id = min(train_data['item_id']) + 1
    num_user = max([max(train_data['user_id']), max(valid_data['user_id']), max(test_data['user_id'])])
    num_item = max([max(train_data['item_id']), max(valid_data['item_id']), max(test_data['item_id'])])

    item_emb_all = list()
    item_emb = None
    if argument.mode == 'svdtrain':
        item_emb_all.append(np.array([[0]*argument.hidden_units]))
        for name in datasets:
            train_data_tmp = pd.read_csv(os.path.join(data_dir, name + '_' + 'train.tsv'), sep='\t')
            valid_data_tmp = pd.read_csv(os.path.join(data_dir, name + '_' + 'valid.tsv'), sep='\t')
            test_data_tmp = pd.read_csv(os.path.join(data_dir, name + '_' + 'test.tsv'), sep='\t')
            num_user_tmp = max([max(train_data_tmp['user_id']), max(valid_data_tmp['user_id']), max(test_data_tmp['user_id'])])
            num_item_tmp = max([max(train_data_tmp['item_id']), max(valid_data_tmp['item_id']), max(test_data_tmp['item_id'])])
            users = np.array(train_data_tmp['user_id'].tolist())
            items = np.array(train_data_tmp['item_id'].tolist())
            ratings = np.array(train_data_tmp['Rating'].tolist())
            interaction_matrix = csr_matrix((ratings, (items, users)), shape=(num_item_tmp+1, num_user_tmp+1)).toarray()
            print(num_item_tmp+1, num_user_tmp+1)
            interaction_matrix = interaction_matrix.astype(float)
            #U, S, Vh = np.linalg.svd(interaction_matrix, full_matrices=True)
            item_emb_, S, user_emb = svds(interaction_matrix, k=argument.hidden_units)
            item_emb_all.append(item_emb_[1:])
        item_emb = np.concatenate(item_emb_all)

    dataset_partitons = {}
    grouped = train_data.groupby('DatasetName')
    mean_values = grouped['item_id'].unique()
    for key, name in enumerate(mean_values.index.tolist()):
        dataset_partitons[dataset_dict[name]] = [min(set(mean_values[key])), max(set(mean_values[key]))]
    softmax_embedding = get_softmax_embedding(dataset_partitons, num_item+1).to(device)

    params = DotDict({
        'learningRate': argument.learning_rate,
        'loss': 'CrossEntropyLoss',
        'maxLength': argument.seq_length,
        'k': argument.eval_k,
        'batchSize': argument.batch_size,
        'fewshotSize': argument.fewshot_size,
        'maskProb': argument.mask_prob,
        'numHeader': argument.num_header,
        'intermediateDim': argument.intermediate_dim,
        'hidden_units': argument.hidden_units,
        'numLayer': argument.num_layer,
        'num_users': num_user, 'num_items': num_item,
        'mode':argument.mode,
        'model':argument.model,
        'dropout_rate': 0,
        'dataset_count': len(dataset_partitons.keys()),
        'global_rank': global_rank,
        'num_gpus': num_gpus,
        'val_step' : argument.val_step,
        'save_step': argument.save_step,
        'action' : argument.action,
        'init_model' : argument.init_model_path
    })
    print(params)
    
    if argument.model == 'recppt':
        model = RecPPT(num_user, num_item+1, params, device=device)

    train_set = aggregate_items(train_data, pad_id=pad_token_id, length=params.maxLength, dataset_dict=dataset_dict)
    if argument.model == 'recppt':
        iterator = GPT2DataIterator(train_set, max_len=params.maxLength, max_item=num_item, mask_prob=params.maskProb, mask_id=pad_token_id,
                                pad_id=pad_token_id, device=device)
        train_dataloader = DataLoader(iterator, batch_size=params.batchSize, shuffle=False, pin_memory=False, collate_fn=personl_define_collate_func)

    valid_dataloaders = dict()
    for name in datasets:
        valid_set = {}
        per_test_data = valid_data[valid_data['DatasetName'] == name]
        for _, row in per_test_data.iterrows():
            valid_set[row['user_id']] = {
                'context': train_set[row['user_id']][0], 'DatasetName': train_set[row['user_id']][1],'negative_sample': eval(row['negative_sample'])
            }
        if argument.model == 'recppt':
            valid_iterator = GPT2TestDataIterator(valid_set, max_len=params.maxLength, mask_id=mask_token_id, pad_id=pad_token_id, device=device) 
            valid_dataloader = DataLoader(valid_iterator, batch_size=params.batchSize*2, pin_memory=False, shuffle=False,collate_fn=personl_define_collate_func_test)
        valid_dataloaders[name] = valid_dataloader

    test_dataloaders = dict()
    for name in datasets:
        test_set = {}
        per_test_data = test_data[test_data['DatasetName'] == name]
        for _, row in per_test_data.iterrows():
            test_set[row['user_id']] = {
                'context': train_set[row['user_id']][0], 'DatasetName': train_set[row['user_id']][1], 'negative_sample': eval(row['negative_sample'])
            }
        if argument.model == 'recppt':
            test_iterator = GPT2TestDataIterator(test_set, max_len=params.maxLength, mask_id=mask_token_id, pad_id=pad_token_id, device=device)
            test_dataloader = DataLoader(test_iterator, batch_size=params.batchSize*3, pin_memory=False, shuffle=False, collate_fn=personl_define_collate_func_test)
        test_dataloaders[name] = test_dataloader

    metrics = [nDCG(), RecallAtK()]
    checkpoint_path ='./exps_' + params.mode
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, datasets_name)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

    AdamOptimizer = DeepSpeedCPUAdam if argument.offload else FusedAdam
    if params.mode == 'train':
        if argument.action == 'fewshot':
            model.reprogramming.init_emb_with_gpt2(model.item_emb.weight)
        grouped_parameters = get_optimizer_grouped_parameters(model, argument.weight_decay)
        
    elif argument.mode == 'freezetrain':
        for i, (name, param) in enumerate(model.named_parameters()):
                if 'wte' in name or 'item_emb' in name or 'gpt2' in name:
                    param.requires_grad = False
        grouped_parameters = get_optimizer_grouped_parameters(model, argument.weight_decay)

    elif argument.mode == 'svdtrain':
        item_emb = torch.Tensor(item_emb).to(device).contiguous()   
        if argument.model == 'recppt':
            model.gpt2.wte = torch.nn.Embedding.from_pretrained(item_emb)
            model.item_emb = torch.nn.Embedding.from_pretrained(item_emb)
            for i, (name, param) in enumerate(model.named_parameters()):
                if 'wte' in name or 'item_emb' in name:
                    param.requires_grad = False
        grouped_parameters = get_optimizer_grouped_parameters(model, argument.weight_decay)

    for n, p in model.named_parameters():
        if n.__contains__("softmax") or not p.requires_grad:
            continue
        p.register_hook(lambda gradient: torch.clamp(gradient, -1.0, 1.0))

    print("fix parameters list:")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            print(n, p.shape)

    AdamOptimizer = DeepSpeedCPUAdam if argument.offload else FusedAdam
    optimizer = AdamOptimizer(grouped_parameters,
                              lr=argument.learning_rate,
                              betas=(0.9, 0.95))
    
    num_update_steps_per_epoch = len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=argument.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=argument.num_warmup_steps,
        num_training_steps=argument.epoch * num_update_steps_per_epoch,
    )

    model, optim, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=argument,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if argument.mode in ['train', 'freezetrain', 'svdtrain']:
        train(
            model, argument.epoch, train_dataloader, valid_dataloaders, test_dataloaders, optim,
            k=params.k, metrics=metrics, params=params, init_item_emb=item_emb, device=device, checkpoint_path=checkpoint_path, softmax_embedding=softmax_embedding
        )
