# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn

from ERC_dataset import MELD_loader
from model import PFE_model
from utils import make_batch_bert

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support
    

def main():    
    initial = 'pretrained'
    model_type='bert-base-cased'
    make_batch = make_batch_bert
    freeze_type = 'no_freeze'    
    sample = 1
    last = False
    
    """Dataset Loading"""
    dataset_list = ['MELD']
    DATA_loader_list = [MELD_loader]
    dataclass = "sentiment"
    dataType = 'multi'
    
    """Log"""
    log_path = os.path.join('test.log')
    fileHandler = logging.FileHandler(log_path)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""
    for dataset, DATA_loader in zip(dataset_list, DATA_loader_list):
        data_path = os.path.join('data/dataset', dataset, dataType)

        save_path = 'models'
        print("###Save Path### ", save_path)
    
        dev_path = os.path.join(data_path, dataset+'_dev.txt')
        test_path = os.path.join(data_path, dataset+'_test.txt')

        dev_dataset = DATA_loader(dev_path, dataclass)
        dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)        

        test_dataset = DATA_loader(test_path, dataclass)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
        
        print('Data: ', dataset, '!!!')       
        model = PFE_model()
        modelfile = os.path.join(save_path, 'model.bin')
        model.load_state_dict(torch.load(modelfile))
        model = model.cuda()    
        model.eval()           

        """Dev & Test evaluation"""
        logger.info('####### ' + dataset + ' #######')
        
        dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

        test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')

        print(test_pre, test_rec, test_fbeta)
        logger.info('Fscore ## accuracy: {}, dev-fscore: {}, test-fscore: {}'.format(test_acc*100, dev_fbeta, test_fbeta))
        logger.info('')
    

def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)
    return acc, pred_list, label_list


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    