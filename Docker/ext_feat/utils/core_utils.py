import pdb

import numpy as np
import torch
from utils.utils import *
import os
from models.model_abmil import ABMIL, MADMIL, MADMIL_CLASS
from models.model_abmil_add import ABMIL_ADD
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_clam_add import CLAM_SB_ADD
from models.model_acmil import ACMIL
from models.model_acmil_add import ACMIL_ADD
from models.model_pool import Max_Pool, Mean_Pool
from models.model_pool_instance import Max_Pool_Ins, Mean_Pool_Ins
from models.model_dtfd import DimReduction, Attention, Classifier_1fc, Attention_with_Classifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import f1_score

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_all = {'y_true':[],'y_pred':[]}

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        # pdb.set_trace()
        if count == 0:
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

    def get_f1(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        f1 = f1_score(y_true,y_pred,average='macro')
        return f1



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class EarlyStopping_DTFD:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    def __call__(self, epoch, val_loss,classifier, attention, dimReduction, attCls, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, classifier, attention, dimReduction, attCls, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, classifier, attention, dimReduction, attCls, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, classifier, attention, dimReduction, attCls, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(classifier.state_dict(), ckpt_name + '0.pt')
        torch.save(attention.state_dict(), ckpt_name + '1.pt')
        torch.save(dimReduction.state_dict(), ckpt_name + '2.pt')
        torch.save(attCls.state_dict(), ckpt_name + '3.pt')
        self.val_loss_min = val_loss



def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    if args.model_type != 'dtfd':
        model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
        if args.model_type == 'clam' and args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.model_size is not None and args.model_type != 'mil':
            model_dict.update({"size_arg": args.model_size})
        
        if args.model_type in ['clam_sb', 'clam_sb_add', 'clam_mb']:
            if args.subtyping:
                model_dict.update({'subtyping': True})
            
            if args.B > 0:
                model_dict.update({'k_sample': args.B})
            
            if args.inst_loss == 'svm':
                from topk.svm import SmoothTop1SVM
                instance_loss_fn = SmoothTop1SVM(n_classes = 2)
                if device.type == 'cuda':
                    instance_loss_fn = instance_loss_fn.cuda()
            else:
                instance_loss_fn = nn.CrossEntropyLoss()

            if args.model_type =='clam_sb':
                model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            elif args.model_type == 'clam_sb_add':
                model = CLAM_SB_ADD(**model_dict, instance_loss_fn=instance_loss_fn)    
            elif args.model_type == 'clam_mb':
                model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
            else:
                raise NotImplementedError   
            
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


        model.relocate()
        print('Done!')
        print_network(model)
            
        print('\nInit optimizer ...', end=' ')
        optimizer = get_optim(model, args)
        print('Done!')   
    elif args.model_type == 'dtfd':
        dimReduction = DimReduction(args.in_chn, args.mDim)
        attention = Attention(args.mDim)
        classifier = Classifier_1fc(args.mDim, args.n_classes, droprate= 0)
        attCls = Attention_with_Classifier(L=args.mDim, num_cls=args.n_classes, droprate=args.drop_out2)
    
        dimReduction.relocate()
        attention.relocate()
        classifier.relocate()
        attCls.relocate()
        print('Done!')
        
        print_network(dimReduction)
        print_network(attention)
        print_network(classifier)
        print_network(attCls)

        trainable_parameters = []
        trainable_parameters += list(classifier.parameters())
        trainable_parameters += list(attention.parameters())
        trainable_parameters += list(dimReduction.parameters())

        print('\nInit optimizers ...', end=' ')
        optimizer0 = optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.reg)
        optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, attCls.parameters()), lr=args.lr, weight_decay=args.reg)
        print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        if args.model_type == 'dtfd':
            early_stopping = EarlyStopping_DTFD(patience = 20, stop_epoch=50, verbose=True)
        else:
            early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None
    print('Done!')
    
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_sb_add', 'clam_mb'] and not args.no_inst_cluster:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,early_stopping, writer, loss_fn, args.results_dir)
        elif args.model_type == 'dtfd':
            train_loop_dtfd(classifier, attention, dimReduction, attCls, optimizer0,optimizer1, args.n_classes, epoch, writer, train_loader, args.distill, loss_fn)
            stop = validate_dtfd(cur, epoch, classifier, attention, dimReduction, attCls, val_loader, args.n_classes,loss_fn, early_stopping, writer, args.results_dir, args.distill) 
        elif args.model_type in ['acmil', 'acmil_add']:
            train_loop_acmil(args, epoch, model, train_loader, optimizer, args.n_classes, args.n_token, writer, loss_fn)
            stop = validate_acmil(args, cur, epoch, model, val_loader, args.n_classes,early_stopping, writer, loss_fn, args.results_dir)    
        elif args.model_type == 'madmil_diff':
            train_loop_MAD(epoch, model, train_loader, optimizer, args.n_classes, args.n, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes,early_stopping, writer, loss_fn, args.results_dir)        
        elif args.model_type == 'madmil_class':
            train_loop_MAD_CLASS(epoch, model, train_loader, optimizer, args.n_classes, args.n, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes,early_stopping, writer, loss_fn, args.results_dir)        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes,early_stopping, writer, loss_fn, args.results_dir)
        if stop: 
            break

    if args.early_stopping:
        if args.model_type == 'dtfd':
            classifier.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint0.pt".format(cur))))
            attention.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint1.pt".format(cur))))
            dimReduction.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint2.pt".format(cur))))
            attCls.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint3.pt".format(cur))))
        else:
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        if args.model_type == 'dtfd':
            torch.save(classifier.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint0.pt".format(cur)))
            torch.save(attention.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint1.pt".format(cur)))
            torch.save(dimReduction.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint2.pt".format(cur)))
            torch.save(attCls.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint3.pt".format(cur)))
        else:
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if args.model_type in ['acmil', 'acmil_add']:
        _, val_error, val_loss, val_auc, val_logger= summary_acmil(args, model, val_loader, args.n_classes, loss_fn)
        print('Val error: {:.4f}, Val loss: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_loss, val_auc))
        
        results_dict, test_error, test_loss, test_auc, acc_logger = summary_acmil(args, model, test_loader, args.n_classes, loss_fn)
        print('Test error: {:.4f}, Test loss: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_loss, test_auc))
    elif args.model_type == 'dtfd':
        _, val_error, val_loss, val_auc, val_logger= summary_dtfd(classifier, attention, dimReduction, attCls, val_loader, args.n_classes, loss_fn, args.distill)
        print('Val error: {:.4f}, Val loss: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_loss, val_auc))
        
        results_dict, test_error, test_loss, test_auc, acc_logger = summary_dtfd(classifier, attention, dimReduction, attCls, test_loader, args.n_classes, loss_fn, args.distill)
        print('Test error: {:.4f}, Test loss: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_loss, test_auc))
    else:
        _, val_error, val_loss, val_auc, val_logger= summary(model, val_loader, args.n_classes, loss_fn)
        print('Val error: {:.4f}, Val loss: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_loss, val_auc))
        
        results_dict, test_error, test_loss, test_auc, acc_logger = summary(model, test_loader, args.n_classes, loss_fn)
        print('Test error: {:.4f}, Test loss: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_loss, test_auc))
        
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    test_f1 = acc_logger.get_f1()
    val_f1 = val_logger.get_f1()

    if writer:
        writer.add_scalar('final/val_f1', val_f1, 0)
        writer.add_scalar('final/val_loss', val_loss, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_f1', test_f1, 0)
        writer.add_scalar('final/test_loss', test_loss, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
        
    return results_dict, test_auc, val_auc, test_f1, val_f1, test_loss, val_loss


def train_loop_acmil(args, epoch, model, loader, optimizer, n_classes, n_token= 1, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_loss_slide= 0.
    train_loss_sub= 0.
    train_loss_diff= 0.
    
    train_error = 0.
    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        sub_preds, slide_preds, attn = model(data,use_attention_mask=True)
        if args.model_type == 'acmil_add':
            slide_preds = torch.sum(slide_preds, dim=0, keepdim=True) # 1x2

        if n_token > 1:
            loss0 = loss_fn(sub_preds, label.repeat_interleave(n_token))
        else:
            loss0 = torch.tensor(0.)
        loss1 = loss_fn(slide_preds, label)

        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        attn = torch.softmax(attn, dim=-1)
        for i in range(int(n_token)):
            for j in range(i + 1, n_token):
                diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                            n_token * (n_token - 1) / 2)

        loss = diff_loss + loss0 + loss1
        
        Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]
        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        train_loss_slide += loss1.item()
        train_loss_sub += loss0.item()
        train_loss_diff += diff_loss.item()

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_loss_slide /= len(loader)
    train_loss_sub /= len(loader)
    train_loss_diff /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_loss_slide: {:.4f}, train_loss_sub: {:.4f}, train_loss_diff: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_loss_slide, train_loss_sub, train_loss_diff, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, Att_scores, instance_dict = model(data, label=label, instance_eval=True)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + 0.1 * instance_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop_dtfd(classifier, attention, dimReduction, UClassifier, optimizer0, optimizer1, n_classes, epoch, writer, loader, distill, criterion):
    
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    total_instance= 4
    numGroup= 4

    # Set the network to training mode
    dimReduction.train()
    attention.train()
    classifier.train()
    UClassifier.train()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss0 = 0.
    train_loss1 = 0.

    train_error = 0.
    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        tfeat_tensor = data.to(device, dtype=torch.float32)
        tslideLabel = label.to(device)

        instance_per_group = total_instance // numGroup
        feat_index = torch.randperm(tfeat_tensor.shape[0]).to(device)
        index_chunk_list = torch.tensor_split(feat_index, numGroup)


        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []

        for tindex in index_chunk_list:
            slide_sub_labels.append(tslideLabel)
            subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=tindex)
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

        ## optimization for the first tier
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  ### numGroup
        loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
        loss_value0 = loss0.item()
        train_loss0 += loss_value0
        optimizer0.zero_grad()
        loss0.backward(retain_graph=True)
        optimizer0.step()

        ## optimization for the second tier
        gSlidePred = UClassifier(slide_pseudo_feat)
        loss1 = criterion(gSlidePred, tslideLabel).mean()
        loss_value1 = loss1.item()
        train_loss1 += loss_value1
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        error = calculate_error(torch.topk(gSlidePred, 1, dim = 1)[1], tslideLabel)
        train_error += error
        acc_logger.log(torch.topk(gSlidePred, 1, dim = 1)[1], tslideLabel)

    # calculate loss and error for epoch
    train_loss0 /= len(loader)
    train_loss1 /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss0: {:.4f}, train_loss1: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss0, 
                                                                                     train_loss1, train_error))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss0', train_loss0, epoch)
        writer.add_scalar('train/loss1', train_loss1, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.

    train_error = 0.
    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        logits, Y_prob, Y_hat, _, _ = model(data)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)       
        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def train_loop_MAD(epoch, model, loader, optimizer, n_classes, n= 1,  writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_loss_slide= 0.
    train_loss_diff= 0.
    
    train_error = 0.
    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        logits, Y_prob, Y_hat, _, A = model(data)
        acc_logger.log(Y_hat, label)
        loss0 = loss_fn(logits, label)       

        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        for i in range(int(n)):
            for j in range(i + 1, n):
                diff_loss += torch.cosine_similarity(A[:, i], A[:, j], dim=-1).mean() / (
                            n * (n - 1) / 2)

        loss = diff_loss + loss0
        train_loss += loss.item()
        train_loss_slide += loss0.item()
        train_loss_diff += diff_loss.item()

        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_loss_slide /= len(loader)
    train_loss_diff /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_loss_slide: {:.4f}, train_loss_diff: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_loss_slide, train_loss_diff, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def train_loop_MAD_CLASS(epoch, model, loader, optimizer, n_classes, n= 1,  writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_loss_slide= 0.
    train_loss_class= 0.
    
    train_error = 0.
    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        logits, Y_prob, Y_hat, _, sub_preds = model(data)
        acc_logger.log(Y_hat, label)
        loss0 = loss_fn(logits, label)       
        if n > 1:
            loss1 = loss_fn(sub_preds, torch.arange(n).to(device))
        else:
            loss1 = torch.tensor(0.)

        loss = loss1 + loss0
        train_loss += loss.item()
        train_loss_slide += loss0.item()
        train_loss_class += loss1.item()

        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_loss_slide /= len(loader)
    train_loss_class /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_loss_slide: {:.4f}, train_loss_class: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_loss_slide, train_loss_class, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)



def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            logits, Y_prob, Y_hat, _, _ = model(data)
                
            acc_logger.log(Y_hat, label) 
            loss = loss_fn(logits, label)
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
      
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False




def validate_acmil(args, cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            sub_preds, slide_preds, attn = model(data,use_attention_mask= False)
            if args.model_type == 'acmil_add':
                slide_preds = torch.sum(slide_preds, dim=0, keepdim=True) # 1x2

            Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]   
            Y_prob = F.softmax(slide_preds, dim = 1) 
            acc_logger.log(Y_hat, label) 
            loss = loss_fn(slide_preds, label)
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
      
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False




def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()
            # pdb.set_trace()
            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_dtfd(cur, epoch, classifier, attention, dimReduction, UClassifier,loader, n_classes, criterion, early_stopping, writer, results_dir, distill):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_instance= 4
    numGroup= 4

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    instance_per_group = total_instance // numGroup

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            tfeat = data.to(device, dtype=torch.float32)
            tslideLabel = label.to(device)

            midFeat = dimReduction(tfeat)

            AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

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
            Y_prob = allSlide_pred_softmax

            loss = criterion(allSlide_pred_softmax, tslideLabel)
            acc_logger.log(Y_hat, tslideLabel)
            val_loss += loss.item()

            error = calculate_error(Y_hat, tslideLabel)
            val_error += error

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = tslideLabel.item()

        val_error /= len(loader)
        val_loss /= len(loader)
        
        if n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
        
        else:
            auc = roc_auc_score(labels, prob, multi_class='ovr')
        
        if writer:
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/auc', auc, epoch)
            writer.add_scalar('val/error', val_error, epoch)

        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

        if early_stopping:
            assert results_dir
            early_stopping(epoch, val_loss, classifier, attention, dimReduction, UClassifier, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint".format(cur)))
            
            if early_stopping.early_stop:
                print("Early stopping")
                return True

        return False


def summary_acmil(args, model, loader, n_classes, loss_fn):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
       
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            sub_preds, slide_preds, attn = model(data, use_attention_mask=False)
            if args.model_type == 'acmil_add':
                slide_preds = torch.sum(slide_preds, dim=0, keepdim=True) # 1x2
                    
        Y_hat = torch.topk(slide_preds, 1, dim = 1)[1]   
        Y_prob = F.softmax(slide_preds, dim = 1) 
            
        loss = loss_fn(slide_preds, label)
        test_loss += loss.item()
                        
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error
        
    test_loss /= len(loader)
    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, test_loss, auc, acc_logger

    
def summary(model, loader, n_classes, loss_fn):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        # pdb.set_trace()
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        loss = loss_fn(logits, label)
        test_loss += loss.item()
                        
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error
        
    test_loss /= len(loader)
    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, test_loss, auc, acc_logger



def summary_dtfd(classifier, attention, dimReduction, UClassifier, loader, n_classes, criterion, distill):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()
    
    total_instance= 4
    numGroup= 4
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    instance_per_group = total_instance // numGroup

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            tfeat = data.to(device, dtype=torch.float32)
            tslideLabel = label.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            midFeat = dimReduction(tfeat)

            AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

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

            loss = criterion(allSlide_pred_softmax, tslideLabel)
            test_loss += loss.item()

            error = calculate_error(Y_hat, tslideLabel)
            test_error += error
                        
            acc_logger.log(Y_hat, tslideLabel)
            probs = allSlide_pred_softmax.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = tslideLabel.item()
        
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': tslideLabel.item()}})
        
    test_loss /= len(loader)
    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, test_loss, auc, acc_logger