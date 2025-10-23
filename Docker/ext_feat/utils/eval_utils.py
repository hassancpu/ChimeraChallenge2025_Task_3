import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_abmil import ABMIL, MADMIL, MADMIL_CLASS
from models.model_abmil_add import ABMIL_ADD
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_clam_add import CLAM_SB_ADD
from models.model_acmil import ACMIL
from models.model_acmil_add import ACMIL_ADD
from models.model_pool import Max_Pool, Mean_Pool
from models.model_pool_instance import Max_Pool_Ins, Mean_Pool_Ins
from models.model_dtfd import DimReduction, Attention, Classifier_1fc, Attention_with_Classifier
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    if args.model_type =='clam_sb_add':
        model = CLAM_SB_ADD(**model_dict)    
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'abmil':
        model = ABMIL(**model_dict) 
    elif args.model_type == 'abmil_add':
        model = ABMIL_ADD(**model_dict)     
    elif args.model_type == 'acmil':
        model = ACMIL(**model_dict, n_token= args.n_token, n_masked_patch= args.n_masked_patch)   
    elif args.model_type == 'acmil_add':
            model = ACMIL_ADD(**model_dict, n_token= args.n_token, n_masked_patch= args.n_masked_patch)            
    elif args.model_type == 'madmil':
        model = MADMIL(**model_dict, n= args.n, head_size= args.head_size)     
    elif args.model_type == 'madmil_diff':
        model = MADMIL(**model_dict, n= args.n, head_size= args.head_size)  
    elif args.model_type == 'madmil_class':
        model = MADMIL_CLASS(**model_dict, n= args.n, head_size= args.head_size)                             
    elif args.model_type == 'max_pool':
        model = Max_Pool(**model_dict)   
    elif args.model_type == 'max_pool_ins':
        model = Max_Pool_Ins(**model_dict)           
    elif args.model_type == 'mean_pool':
        model = Mean_Pool(**model_dict) 
    elif args.model_type == 'mean_pool_ins':
        model = Mean_Pool_Ins(**model_dict)               
    elif args.model_type == 'dtfd':
        dimReduction = DimReduction(args.in_chn, args.mDim)
        attention = Attention(args.mDim)
        classifier = Classifier_1fc(args.mDim, args.n_classes, droprate= 0)
        attCls = Attention_with_Classifier(L=args.mDim, num_cls=args.n_classes, droprate=args.drop_out2)

    if args.model_type != 'dtfd':
        print_network(model)
  
        ckpt = torch.load(ckpt_path)
        ckpt_clean = {}
        for key in ckpt.keys():
            if 'instance_loss_fn' in key:
                continue
            ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        model.load_state_dict(ckpt_clean, strict=True)

        model.relocate()
        model.eval()
        return model
    else:
        ckpt_path_dim = ckpt_path.split('.')[0] + '2.' + ckpt_path.split('.')[1]
        ckpt = torch.load(ckpt_path_dim)
        ckpt_clean = {}
        for key in ckpt.keys():
            if 'instance_loss_fn' in key:
                continue
            ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        dimReduction.load_state_dict(ckpt_clean, strict=True)

        dimReduction.relocate()
        dimReduction.eval()

        ckpt_path_att = ckpt_path.split('.')[0] + '1.' + ckpt_path.split('.')[1]
        ckpt = torch.load(ckpt_path_att)
        ckpt_clean = {}
        for key in ckpt.keys():
            if 'instance_loss_fn' in key:
                continue
            ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        attention.load_state_dict(ckpt_clean, strict=True)

        attention.relocate()
        attention.eval()

        ckpt_path_clss = ckpt_path.split('.')[0] + '0.' + ckpt_path.split('.')[1]
        ckpt = torch.load(ckpt_path_clss)
        ckpt_clean = {}
        for key in ckpt.keys():
            if 'instance_loss_fn' in key:
                continue
            ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        classifier.load_state_dict(ckpt_clean, strict=True)

        classifier.relocate()
        classifier.eval()

        ckpt_path_attCls = ckpt_path.split('.')[0] + '3.' + ckpt_path.split('.')[1]
        ckpt = torch.load(ckpt_path_attCls)
        ckpt_clean = {}
        for key in ckpt.keys():
            if 'instance_loss_fn' in key:
                continue
            ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        attCls.load_state_dict(ckpt_clean, strict=True)

        attCls.relocate()
        attCls.eval()

        return dimReduction, attention, classifier, attCls

def eval(dataset, args, ckpt_path):
    if args.model_type == 'dtfd':
        dimReduction, attention, classifier, attCls = initiate_model(args, ckpt_path)
    else:
        model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)

    if args.model_type == 'dtfd':
        patient_results, all_att, test_error, auc, df, acc = summary_dtfd(classifier, attention, dimReduction, attCls, loader, args)
    else:    
        patient_results, all_att, test_error, auc, df, acc = summary(model, loader, args)

    print('acc: ', acc)
    print('auc: ', auc)
    return patient_results, all_att, acc, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    all_att= {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if isinstance(model, (ACMIL,)) or isinstance(model, (ACMIL_ADD,)):
                sub_preds, logits_raw, att = model(data,use_attention_mask=False)
                if isinstance(model, (ACMIL_ADD,)):
                    slide_preds = torch.sum(logits_raw, dim=0, keepdim=True) # 1x2
                    Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]   
                    Y_prob = F.softmax(slide_preds, dim = 1) 
                else:
                    Y_prob = F.softmax(logits_raw, dim = 1) 
                    Y_hat = torch.topk(logits_raw, 1, dim = 1)[1]    
            else:
                logits, Y_prob, Y_hat, logits_raw, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        if Y_hat.item() == label.item():
            if isinstance(model, (ACMIL,)) or isinstance(model, (ACMIL_ADD,)):
                all_att.update({slide_id: [logits_raw.cpu().numpy(), Y_hat.cpu().numpy(), att.cpu().numpy()]})
            else:    
                all_att.update({slide_id: [logits_raw.cpu().numpy(), Y_hat.cpu().numpy(), results_dict['attention'].cpu().numpy()]})

        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, all_att, test_error, auc_score, df, acc_logger.get_f1()

def summary_dtfd(classifier, attention, dimReduction, UClassifier, loader, args, distill= 'AFS'):    
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()
    
    total_instance= 4
    numGroup= 4
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    instance_per_group = total_instance // numGroup

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    all_att= {}
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            tfeat = data.to(device, dtype=torch.float32)
            tslideLabel = label.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            midFeat = dimReduction(tfeat)

            AA = attention(midFeat, isNorm=False).squeeze(0)  ## N
            patch_logits = torch.zeros((midFeat.shape[0], args.n_classes)).to(device)
            feat_index = torch.randperm(tfeat.shape[0]).to(device)
            index_chunk_list = torch.tensor_split(feat_index, numGroup)

            slide_d_feat = []


            for tindex in index_chunk_list:
                tmidFeat = midFeat.index_select(dim=0, index=tindex)

                tAA = AA.index_select(dim=0, index=tindex)
                tAA = torch.softmax(tAA, dim=0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                patch_logits.index_copy_(0, tindex, patch_pred_logits)

                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                if distill == 'MaxMinS':
                    topk_idx_max = sort_idx[:instance_per_group].long()
                    topk_idx_min = sort_idx[-instance_per_group:].long()
                    topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                    d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    slide_d_feat.append(d_inst_feat)
                elif distill == 'MaxS':
                    topk_idx_max = sort_idx[:instance_per_group].long()
                    topk_idx = topk_idx_max
                    d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    slide_d_feat.append(d_inst_feat)
                elif distill == 'AFS':
                    slide_d_feat.append(tattFeat_tensor)

            slide_d_feat = torch.cat(slide_d_feat, dim=0)

            gSlidePred = UClassifier(slide_d_feat)
            allSlide_pred_softmax = torch.softmax(gSlidePred, dim=1)
            Y_hat = torch.topk(gSlidePred, 1, dim = 1)[1]

            error = calculate_error(Y_hat, tslideLabel)
            test_error += error
                        
            acc_logger.log(Y_hat, tslideLabel)
            probs = allSlide_pred_softmax.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = tslideLabel.item()
        
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': tslideLabel.item()}})
            if Y_hat.item() == label.item():
                all_att.update({slide_id: [patch_logits.cpu().numpy(), Y_hat.cpu().numpy(), AA.cpu().numpy()]})

    test_error /= len(loader)

    if args.n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        for class_idx in range(args.n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    df = pd.DataFrame(patient_results)
    return patient_results, all_att, test_error, auc, df, acc_logger.get_f1()