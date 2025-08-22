
import numpy as np
import torch
from utils.utils import *
import os
import torch.nn.functional as F
from models.model_abmil import ABMIL_Surv, ABMIL_Surv_PG, ABMIL_Surv_Img, ABMIL_Surv_ImgCli, ABMIL_Surv_ImgRNA, ABMIL_Surv_RNACli
from lifelines.utils import concordance_index


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

        score = val_loss

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
        '''Saves model when C-index increases.'''
        if self.verbose:
            print(f'C-index increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
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
        from tensorboardX import SummaryWriter # type: ignore
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
    loss_fn = CoxSurvLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"feat_type": args.feat_type}    
    if args.model_type == 'abmil':
        model = ABMIL_Surv(**model_dict)
    elif args.model_type == 'pg':
        model =  ABMIL_Surv_PG(**model_dict)
    elif args.model_type == 'img':
        model =  ABMIL_Surv_Img(**model_dict)
    elif args.model_type == 'imgcli':
        model =  ABMIL_Surv_ImgCli(**model_dict)
    elif args.model_type == 'imgrna':
        model =  ABMIL_Surv_ImgRNA(**model_dict)
    elif args.model_type == 'rnacli':
        model =  ABMIL_Surv_RNACli(**model_dict)

    print_network(model)
    model.relocate()
    print('Done!')
        
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True,weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience= 30, stop_epoch= 50, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, early_stopping, writer, loss_fn, args.results_dir)
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
        
    _, val_cindex, val_loss = summary(model, val_loader, loss_fn)
    print('Val c-Index: {:.4f}, , Val loss: {:.4f}'.format(val_cindex, val_loss))

    results_dict, test_cindex, test_loss = summary(model, test_loader, loss_fn)
    print('Test c-Index: {:.4f}, Test loss: {:.4f}'.format(test_cindex, test_loss))

    if writer:
        writer.add_scalar('final/val_loss', val_loss, 0)
        writer.add_scalar('final/val_cindex', val_cindex, 0)
        writer.add_scalar('final/test_loss', test_loss, 0)
        writer.add_scalar('final/test_cindex', test_cindex, 0)
        writer.close()
            
    return results_dict, test_cindex, val_cindex, test_loss, val_loss


# Training loop for all models
def train_loop(epoch, model, loader, optimizer, writer=None, loss_fn=None, gc=1):      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0

    print('\nEpoch {}:'.format(epoch))
    all_risk_scores = []
    all_times = []
    all_events = []

    optimizer.zero_grad()

    for batch_idx, (feature_list, clinic, rna, time, event) in enumerate(loader):
        batch_risks = []

        for i in range(len(feature_list)):
            feats = feature_list[i].to(device)          # [N_patches, feat_dim]
            clin = clinic[i].to(device)                 # [1, clinical_dim]
            rna_ = rna[i].unsqueeze(0).to(device)       # [1, rna_dim]

            risk, _ = model(feats, clin, rna_)          # scalar per WSI
            batch_risks.append(risk)

        risk_scores = torch.cat(batch_risks).squeeze()  # shape: [batch_size]
        time = time.to(device)
        event = event.to(device)

        # Calculate loss
        risk_scores = risk_scores - risk_scores.max(dim=0, keepdim=True).values
        loss = loss_fn(risk_scores, time, event)
        loss_value = loss.item()
        train_loss += loss_value

        # Store for C-index
        all_risk_scores.append(risk_scores.detach().cpu())
        all_times.append(time.detach().cpu())
        all_events.append(event.detach().cpu())

        loss = loss / gc
        loss.backward()

        if (batch_idx + 1) % gc == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    train_loss /= len(loader)

    all_risk_scores = torch.cat(all_risk_scores).numpy()
    all_times = torch.cat(all_times).numpy()
    all_events = torch.cat(all_events).numpy()

    # Calculate C-index
    c_index = concordance_index(all_times, -all_risk_scores, all_events)

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


# Validation loop for all models
def validate(cur, epoch, model, loader, early_stopping=None, writer=None, loss_fn=None, results_dir=None):     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    val_loss = 0.
    all_risk_scores = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for _, (feature_list, clinic, rna, time, event) in enumerate(loader):
            batch_risks = []

            for i in range(len(feature_list)):
                feats = feature_list[i].to(device)
                clin = clinic[i].to(device)
                rna_ = rna[i].unsqueeze(0).to(device)

                risk, _ = model(feats, clin, rna_)
                batch_risks.append(risk)
                
            risk_scores = torch.cat(batch_risks).squeeze()  # shape: [batch_size]

            time = time.to(device)
            event = event.to(device)

            loss = loss_fn(risk_scores, time, event)
            val_loss += loss.item()

            all_risk_scores.append(risk_scores.detach().cpu())
            all_times.append(time.detach().cpu())
            all_events.append(event.detach().cpu())

    val_loss /= len(loader)

    all_risk_scores = torch.cat(all_risk_scores).numpy()
    all_times = torch.cat(all_times).numpy()
    all_events = torch.cat(all_events).numpy()

    c_index = concordance_index(all_times, -all_risk_scores, all_events)

    print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, c_index))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c_index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False


# Summary for all models
def summary(model, loader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_loss = 0
    all_risk_scores = []
    all_times = []
    all_events = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    with torch.no_grad():
        for batch_idx, (feature_list, clinic, rna, time, event) in enumerate(loader):
            batch_risks = []

            for i in range(len(feature_list)):
                feats = feature_list[i].to(device)
                clin = clinic[i].to(device)
                rna_ = rna[i].unsqueeze(0).to(device)

                risk, _ = model(feats, clin, rna_)
                batch_risks.append(risk)

            risk_scores = torch.cat(batch_risks).squeeze()
            time = time.to(device)
            event = event.to(device)

            loss = loss_fn(risk_scores, time, event)
            test_loss += loss.item()

            # Store C-index data
            all_risk_scores.append(risk_scores.detach().cpu())
            all_times.append(time.detach().cpu())
            all_events.append(event.detach().cpu())

            # Store results per patient (assumes 1:1 WSI-patient)
            batch_size = risk_scores.shape[0]
            start_idx = batch_idx * batch_size

            for j in range(batch_size):
                sid = slide_ids.iloc[start_idx + j]
                patient_results[sid] = {
                    'slide_id': sid,
                    'risk': risk_scores[j].item(),
                    'survival': time[j].item(),
                    'event': event[j].item()
                }

    test_loss /= len(loader)
    all_risk_scores = torch.cat(all_risk_scores).numpy() 
    all_times = torch.cat(all_times).numpy()
    all_events = torch.cat(all_events).numpy()

    c_index = concordance_index(all_times, -all_risk_scores, all_events)
    return patient_results, c_index, test_loss
