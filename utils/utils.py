import torch
import numpy as np
import torch.nn as nn

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler
import torch.optim as optim
import math
from itertools import islice
import collections
import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set the device for PyTorch
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClinicalFeatureExtractor:
    def __init__(self, numeric_cols=None, categorical_cols=None):
        self.numeric_cols = numeric_cols or ["age", "no_instillations"]
        self.categorical_cols = categorical_cols or [
            "sex", "smoking", "tumor", "stage", "substage", "grade",
            "reTUR", "LVI", "variant", "EORTC"
        ]
        self.preprocessor = None

    def fit(self, clinical_dir, train_slide_ids):
        """
        Fit the preprocessor using only training JSON files.
        
        Args:
            clinical_dir (str): Path to the directory containing clinical JSON files.
            train_slide_ids (list[str]): List of training slide IDs, e.g., ["2A_001_HE", "2A_003_HE"].
        """
        data = []
        for slide_id in train_slide_ids:
            slide_base = slide_id.replace("_HE", "_CD.json")  # convert to your clinical file naming
            json_path = os.path.join(clinical_dir, slide_base)
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data.append(json.load(f))
            else:
                print(f"Warning: Clinical file not found: {json_path}")

        df = pd.DataFrame(data)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
            ]
        )
        self.preprocessor.fit(df)
        return self

    def transform(self, clinical_json_path):
        """Transform a single clinical JSON file into a normalized feature tensor."""
        clinic_raw = json.load(open(clinical_json_path, "r"))
        df = pd.DataFrame([clinic_raw])
        features = self.preprocessor.transform(df)
        features = features.toarray() if hasattr(features, "toarray") else features
        return features

    def save(self, path):
        """Save the fitted preprocessor to disk."""
        joblib.dump((self.numeric_cols, self.categorical_cols, self.preprocessor), path)

    def load(self, path):
        """Load the fitted preprocessor from disk."""
        self.numeric_cols, self.categorical_cols, self.preprocessor = joblib.load(path)
        return self

# def cox_loss(risk, time, event, eps=1e-8):
#     if event.sum() == 0:
#         return torch.tensor([0.0], device=risk.device, requires_grad=True)

#     sorted_idx = torch.argsort(time, descending=True)
#     risk = risk[sorted_idx]
#     event = event[sorted_idx]

#     hazard_ratio = torch.exp(risk)
#     log_cumsum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0) + eps)
#     diff = risk - log_cumsum_hazard
#     loss = -torch.sum(diff * event) / (torch.sum(event) + eps)
#     return loss


def cox_loss(risk, time, event, eps=1e-8):
	risk = risk.view(-1)
	event = event.view(-1).float()
	time = time.view(-1)

	# Sort by descending time
	order = torch.argsort(time, descending=True)
	risk = risk[order]
	event = event[order]

	# Log cumulative hazard
	hazard_ratio = torch.exp(risk)
	log_cum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0) + eps)
	diff = risk - log_cum_hazard
	weights = event / (event.sum() + eps)

	loss = -torch.sum(diff * weights) / (torch.sum(weights) + eps)
	return loss




class CoxSurvLoss(object):
	def __init__(self):
		"""
		Cox proportional hazards loss function.
		This is used for survival analysis tasks.
		"""
		pass
	def __call__(self, risk, time, event):
		return cox_loss(risk, time, event)
		

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
    features = [item[0] for item in batch]                  # list of [N_patches, feat_dim]
    clinicals = torch.stack([item[1] for item in batch])    # [B, C]
    rnas = torch.stack([item[2] for item in batch])         # [B, R]
    times = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    events = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    return features, clinicals, rnas, times, events


def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=2, num_workers=1):
	kwargs = {'num_workers': 6, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 6} if device.type == "cuda" else {}
	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(split_dataset)
			loader = DataLoader(split_dataset, batch_size=2, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
		else:
			loader = DataLoader(split_dataset, batch_size=2, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	else:
		loader = DataLoader(split_dataset, batch_size=2, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)

	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
    for m in module.modules():
        if hasattr(m, 'weight') and m.weight is not None:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

