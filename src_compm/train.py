from model import PFE_model
import torch.nn as nn
import torch
from transformers import get_linear_schedule_with_warmup
from ERC_dataset import MELD_loader
from torch.utils.data import DataLoader
from utils import make_batch_bert
import os, logging
from sklearn.metrics import precision_recall_fscore_support

def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val


def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens, batch_labels
            
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
        

def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))


def get_datasets() :
    dataType = 'multi'
    dataset='MELD'
    data_path = 'data/dataset/MELD/'+dataType+'/'

    train_path = data_path + dataset+'_train.txt'
    dev_path = data_path + dataset+'_dev.txt'
    test_path = data_path + dataset+'_test.txt'

    DATA_loader = MELD_loader
    dev_dataset = DATA_loader(dev_path, 'sentiment')
    train_dataset = DATA_loader(train_path, 'sentiment')
    test_dataset = DATA_loader(test_path, 'sentiment')

    return dev_dataset,train_dataset,test_dataset

logger = logging.getLogger(__name__)
save_path='models'


def fit_model(lr_opt=1e-6,steps=10000,nb_loops=5,eval_average='weighted') :

    dev_dataset,train_dataset,test_dataset=get_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=1,
        shuffle=True, num_workers=2, collate_fn=make_batch_bert)
    train_sample_num = int(len(train_dataloader))

    dev_dataloader = DataLoader(dev_dataset, batch_size=1,
     shuffle=False, num_workers=2, collate_fn=make_batch_bert)

    test_dataloader = DataLoader(test_dataset, batch_size=1,
     shuffle=False, num_workers=2, collate_fn=make_batch_bert)
    
    
    model=PFE_model()
    model=model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.train_params, lr=lr_opt) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps, num_training_steps=steps)
    best_dev_fscore=0
    for epoch in range(nb_loops):
        model.train() 
        print(epoch)
        for i_batch, data in enumerate(train_dataloader):
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break
            
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens, batch_labels
            
            pred_logits = model(batch_input_tokens, batch_speaker_tokens)

            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, batch_labels)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        """Dev & Test evaluation"""
        model.eval()
        
        dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average=eval_average)

        """Best Score & Model Save"""
        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
                
            test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average=eval_average)                
            print(test_pre,test_rec,test_fbeta)
            best_epoch = epoch
            _SaveModel(model, save_path)
        
        
        logger.info('Epoch: {}'.format(epoch))
            
        logger.info('Devleopment ## accuracy: {}, precision: {}, recall: {}, fscore: {}'.format(dev_acc, dev_pre, dev_rec, dev_fbeta))
        logger.info('')
        
    
    logger.info('Final Fscore ## test-accuracy: {}, test-fscore: {}, test_epoch: {}'.format(test_acc, test_fbeta, best_epoch))  


if __name__ == '__main__':
    fit_model()
