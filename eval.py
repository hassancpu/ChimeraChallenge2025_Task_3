from __future__ import print_function


import argparse
import torch # type: ignore
import os
import pandas as pd # type: ignore
from utils.utils import *
from datasets.dataset_generic import Generic_MIL_Dataset
from utils.eval_utils import *


# Generic settings
parser = argparse.ArgumentParser(description='Configurations for WSI Evaluation')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--results_dir', type=str, default=None, help='(default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None, help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None, help='experiment code to load trained models')
parser.add_argument('--seed', type=int, default=2021,help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds')
parser.add_argument('--k_start', type=int, default=-1, help='start fold')
parser.add_argument('--k_end', type=int, default=5, help='end fold')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--splits_dir', type=str, default=None,help='manually specify the set of splits to use')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['bladder-brs'],help='task to train')
parser.add_argument('--model_type', type=str, choices=['abmil', 'pg'], default='abmil', help='type of model')
parser.add_argument('--feat_type', type=str, choices=['uni', 'gigapath'], default='uni', help='type of features to use')

args = parser.parse_args()

args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.task = "bladder-rec"
args.splits_dir = "./splits/bladder-rec"
args.csv_path = './dataset_csv/bladder-rec.csv'
args.results_dir = './results/UNI/MULTI_10x/'
args.eval_dir = './eval_results/UNI/MULTI_10x/'
args.data_root_dir = "/home/20215294/Data/CHIM_Rec/CHIM_Rec_ostu_10x/"
args.feat_type = 'uni'
sub_feat_dir = 'feat_uni'
args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)


settings = {
            'model_type': args.model_type,
            'sub_feat_dir': sub_feat_dir,}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

   
if args.task == 'bladder-rec': 
    args.k = 1
    args.k_end = 1  
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= os.path.join(args.data_root_dir, sub_feat_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='Progression',
                            label_dict = {0:0, 1:1},
                            patient_strat=False,
                            ignore=[])           
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_cindex = []
    all_loss = []

    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        patient_results, cindex, loss, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])

        all_results.append(patient_results)
        all_cindex.append(cindex)
        all_loss.append(loss)

        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_cindex': all_cindex, 'test_loss': all_loss})

    save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.save_dir, save_name))
    
    "Compute the average and std of the metrics"
    test_loss_ave= np.mean(all_loss)
    test_loss_std= np.std(all_loss)

    test_cindex_ave= np.mean(all_cindex)
    test_cindex_std= np.std(all_cindex)
 
    print('\n\nTest:\n loss ± std: {0:.2f} ± {1:.2f}, cindex ± std: {2:.2f} ± {3:.2f}\n\n'.format(test_loss_ave, test_loss_std, test_cindex_ave, test_cindex_std))    