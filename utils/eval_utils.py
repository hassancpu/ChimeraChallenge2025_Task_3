import numpy as np

import torch
import pandas as pd
import torch.nn.functional as F
from models.model_abmil import ABMIL_Surv, ABMIL_Surv_PG
from utils.utils import *
from lifelines.utils import concordance_index


def initiate_model(args, ckpt_path):
    print('\nInit Model...', end=' ')
    model_dict = {"feat_type": args.feat_type}    
    if args.model_type == 'abmil':
        model = ABMIL_Surv(**model_dict)
    elif args.model_type == 'pg':
        model =  ABMIL_Surv_PG(**model_dict)
        
    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    # load the model    
    model.load_state_dict(ckpt_clean, strict=True)
    print('Load checkpoint from {}'.format(ckpt_path))

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)

    print('Init Loader...')
    loader = get_simple_loader(dataset)

    print('\nInit loss function...', end=' ')
    loss_fn = CoxSurvLoss()

    patient_results, c_index, test_loss, df = summary(model, loader, args, loss_fn)
    return patient_results, c_index, test_loss, df

def summary(model, loader, args, loss_fn):
    device = args.device
    model.eval()

    test_loss = 0.0
    all_risk_scores = []
    all_times = []
    all_events = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    counter = 0  # <-- running index for slide_ids

    with torch.no_grad():
        for batch_idx, (feature_list, clinic, rna, time, event) in enumerate(loader):
            batch_risks = []

            # compute risk for each sample in batch
            for i in range(len(feature_list)):
                feats = feature_list[i].to(device)
                clin = clinic[i].to(device)
                rna_ = rna[i].unsqueeze(0).to(device)

                risk, _ = model(feats, clin, rna_)
                batch_risks.append(risk)

            # flatten into 1D vector of risks
            risk_scores = torch.cat(batch_risks, dim=0).view(-1)

            time = time.to(device)
            event = event.to(device)

            loss = loss_fn(risk_scores, time, event)
            test_loss += loss.item()

            # collect for C-index computation
            all_risk_scores.append(risk_scores.detach().cpu())
            all_times.append(time.detach().cpu())
            all_events.append(event.detach().cpu())

            # assign slide_ids in strict sequential order
            for j in range(risk_scores.shape[0]):
                sid = slide_ids.iloc[counter]
                patient_results[sid] = {
                    'slide_id': sid,
                    'risk': risk_scores[j].item(),
                    'survival': time[j].item(),
                    'event': event[j].item()
                }
                counter += 1  # move to next slide_id

    # average loss
    test_loss /= len(loader)

    # concat all collected values
    all_risk_scores = torch.cat(all_risk_scores).view(-1).numpy()
    all_times = torch.cat(all_times).view(-1).numpy()
    all_events = torch.cat(all_events).view(-1).numpy()

    # concordance index: note we negate risk (higher risk = shorter survival)
    c_index = concordance_index(all_times, -all_risk_scores, all_events)

    # build dataframe (now lengths will always match)
    df = pd.DataFrame({
        'slide_id': list(patient_results.keys()),
        'risk': all_risk_scores,
        'time': all_times,
        'event': all_events
    })

    return patient_results, c_index, test_loss, df

