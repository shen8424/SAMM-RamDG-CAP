import warnings
warnings.filterwarnings("ignore")
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
import argparse
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader, collate_fn
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label

from models.RamDG import RamDG

def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust_all(text_input_1, fake_word_pos, extra_text_input_1, extra_indices, device):
    input_ids_remove_SEP = [x[:-1] for x in text_input_1.input_ids]  
    attention_mask_remove_SEP = [x[:-1] for x in text_input_1.attention_mask]
    
    extra_input_ids_remove_CLS_SEP = [x[1:-1] for x in extra_text_input_1.input_ids]
    extra_attention_mask_remove_CLS_SEP = [x[1:-1] for x in extra_text_input_1.attention_mask]
    # fake_token_pos adaptation
    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []
        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist()

        subword_idx = text_input_1.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) 

        for j in fake_word_pos_decimal:
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == j)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    combined_input_ids = []
    combined_attention_mask = []

    for i in range(len(input_ids_remove_SEP)):
        current_input_ids = input_ids_remove_SEP[i]
        current_attention_mask = attention_mask_remove_SEP[i]

        extra_indices_for_current_text = [idx for idx, text_idx in enumerate(extra_indices) if text_idx == i]


        for extra_idx in extra_indices_for_current_text:
            current_input_ids.extend(extra_input_ids_remove_CLS_SEP[extra_idx])
            current_attention_mask.extend(extra_attention_mask_remove_CLS_SEP[extra_idx])
        combined_input_ids.append(current_input_ids)
        combined_attention_mask.append(current_attention_mask)

    maxlen_combined = max([len(x) for x in combined_input_ids])

    combined_input_ids_pad = [x + [0] * (maxlen_combined - len(x)) for x in combined_input_ids]
    combined_attention_mask_pad = [x + [0] * (maxlen_combined - len(x)) for x in combined_attention_mask]

    text_input_1.input_ids = torch.LongTensor(combined_input_ids_pad).to(device)
    text_input_1.attention_mask = torch.LongTensor(combined_attention_mask_pad).to(device)

    return text_input_1, fake_token_pos_batch

def extra_text_input_adjust(text_input1,  device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input1.input_ids]
    maxlen = max([len(x) for x in text_input1.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP]
    text_input1.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)

    attention_mask_remove_SEP = [x[:-1] for x in text_input1.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input1.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    return text_input1

def text_input_adjust(text_input, fake_word_pos, device):
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] 
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)

    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []
        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP)

        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch

  

@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    print_freq = 200

    result_lists = []

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0   

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
    
    TP_all_multicls = np.zeros(4, dtype = int)
    TN_all_multicls = np.zeros(4, dtype = int)
    FP_all_multicls = np.zeros(4, dtype = int)
    FN_all_multicls = np.zeros(4, dtype = int)
    F1_multicls = np.zeros(4)

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, cap_images, image_indices, if_source_name_img, cap_texts, text_indices, if_source_name_text, patch_label) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        
        image = image.to(device,non_blocking=True) 
        
        if len(image_indices) > 0:
            cap_images = cap_images.to(device,non_blocking=True)
        if len(text_indices) > 0:
            text_input = tokenizer(text, max_length=128, padding = False, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)
            cap_texts =  tokenizer(cap_texts, padding = False,max_length=88, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)
            text_input_copy = copy.deepcopy(text_input)
            cap_texts_copy = copy.deepcopy(cap_texts)
            text_input_copy, fake_token_pos = text_input_adjust_all(text_input_copy, fake_word_pos, cap_texts_copy, text_indices, device)
            text_input_orig = extra_text_input_adjust(text_input, device)            
            cap_texts = extra_text_input_adjust(cap_texts, device)
            text_input = text_input_copy
        else:
            text_input = tokenizer(text, max_length=128, padding = False, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)
            text_input, fake_token_pos= text_input_adjust(text_input, fake_word_pos, device)
            text_input_orig = text_input

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, text_input_orig,fake_image_box, fake_token_pos, cap_images, image_indices, if_source_name_img, cap_texts, text_indices, if_source_name_text, is_train=False)

        ##================= real/fake cls ========================## 
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake,dim=1)[:,1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        
        for cls_idx in range(logits_multicls.shape[1]):
            cls_pred = logits_multicls[:, cls_idx]
            cls_pred[cls_pred>=0]=1
            cls_pred[cls_pred<0]=0
            
            TP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 1)).item()
            TN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 0)).item()
            FP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 1)).item()
            FN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 0)).item()
            
        ##================= bbox cls ========================## 
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        IOU_pred.extend(IOU.cpu().tolist())

        IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

        IOU_50_bt[IOU>0.5] = 1
        IOU_75_bt[IOU>0.75] = 1
        IOU_95_bt[IOU>0.95] = 1

        IOU_50.extend(IOU_50_bt.cpu().tolist())
        IOU_75.extend(IOU_75_bt.cpu().tolist())
        IOU_95.extend(IOU_95_bt.cpu().tolist())

        ##================= token cls ========================##  
        token_label = text_input_orig.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
        token_label[token_label==0] = -100 # -100 index = padding token
        token_label[token_label==1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1
        
    
                   
        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)
        

        # F1
        TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()

        token_label_col_count = len(token_label[0])
                
    ##================= real/fake cls ========================## 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AUC_cls = roc_auc_score(y_true, y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    ##================= bbox cls ========================##
    IOU_score = sum(IOU_pred)/len(IOU_pred)
    IOU_ACC_50 = sum(IOU_50)/len(IOU_50)
    IOU_ACC_75 = sum(IOU_75)/len(IOU_75)
    IOU_ACC_95 = sum(IOU_95)/len(IOU_95)
    # ##================= token cls========================##
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    Precision_tok = TP_all / (TP_all + FP_all)
    Recall_tok = TP_all / (TP_all + FN_all)
    F1_tok = 2*Precision_tok*Recall_tok / (Precision_tok + Recall_tok)
    ##================= multi-label cls ========================## 
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
    total_rows, ratio = multi_label_meter.compute_matching_ratio()

            
    for cls_idx in range(logits_multicls.shape[1]):
        Precision_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FP_all_multicls[cls_idx])
        Recall_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FN_all_multicls[cls_idx])
        F1_multicls[cls_idx] = 2*Precision_multicls*Recall_multicls / (Precision_multicls + Recall_multicls)            

    return AUC_cls, ACC_cls, EER_cls, \
        MAP.item(), OP, OR, OF1, CP, CR, CF1, F1_multicls, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok, total_rows, ratio
    
def main_worker(gpu, args, config):
    args.distributed = False
    args.log = True 
    args.device="cuda:0"

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Tokenizer ####
    vocab_file_path = "./vocab.txt"
    tokenizer = BertTokenizerFast(vocab_file=vocab_file_path)

    #### Model ####
    if args.log:
        print(f"Creating RamDG")
    model = RamDG(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True, is_train=False)
    model = model.to(device)

    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir, map_location='cpu')
    state_dict = checkpoint['model']

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    if args.log:
        print('Load checkpoint from %s' % checkpoint_dir)
    msg = model.load_state_dict(state_dict, strict=False)
    if args.log:
        print(msg)

    #### Dataset ####
    if args.log:
        print("Creating dataset")
    _, val_dataset = create_dataset(config)

    val_loader = create_loader(
        [val_dataset],
        [None], 
        batch_size=[config['batch_size_val']],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[collate_fn]
    )[0]

    if args.log:
        print("Start evaluation")

    AUC_cls, ACC_cls, EER_cls, \
    MAP, OP, OR, OF1, CP, CR, CF1, F1_multicls, \
    IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
    ACC_tok, Precision_tok, Recall_tok, F1_tok, total_rows, ratio = evaluation(
        args, model, val_loader, tokenizer, device, config
    )

    val_stats = {
        "AUC_cls": "{:.4f}".format(AUC_cls * 100),
        "ACC_cls": "{:.4f}".format(ACC_cls * 100),
        "EER_cls": "{:.4f}".format(EER_cls * 100),
        "MAP": "{:.4f}".format(MAP * 100),
        "OP": "{:.4f}".format(OP * 100),
        "OR": "{:.4f}".format(OR * 100),
        "OF1": "{:.4f}".format(OF1 * 100),
        "CP": "{:.4f}".format(CP * 100),
        "CR": "{:.4f}".format(CR * 100),
        "CF1": "{:.4f}".format(CF1 * 100),
        "F1_FS": "{:.4f}".format(F1_multicls[0] * 100),
        "F1_FA": "{:.4f}".format(F1_multicls[1] * 100),
        "F1_TS": "{:.4f}".format(F1_multicls[2] * 100),
        "F1_TA": "{:.4f}".format(F1_multicls[3] * 100),
        "IOU_score": "{:.4f}".format(IOU_score * 100),
        "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50 * 100),
        "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75 * 100),
        "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95 * 100),
        "ACC_tok": "{:.4f}".format(ACC_tok * 100),
        "Precision_tok": "{:.4f}".format(Precision_tok * 100),
        "Recall_tok": "{:.4f}".format(Recall_tok * 100),
        "F1_tok": "{:.4f}".format(F1_tok * 100),
        "total_rows": "{:.4f}".format(total_rows)
    }

    eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
    if eval_type == 'test':
        eval_type = 'all'
    log_dir = os.path.join(args.output_dir, args.log_num, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'shell_{eval_type}.txt')
    logger = setlogger(log_file)

    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    log_stats = {
        **{f'val_{k}': v for k, v in val_stats.items()},
        'epoch': args.test_epoch,
    }
    with open(os.path.join(log_dir, f"results_{eval_type}.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint_dir', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--log_num', default='123', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=5)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--test_epoch', default='best', type=str)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main_worker(0, args, config)
