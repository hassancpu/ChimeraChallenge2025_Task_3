
"""
save the coordinates of the features in .pt file
"""

import h5py
import torch
from glob import glob

def save_coord(feat_path, coord_paths):
    feat_paths = glob(feat_path + '/*.h5')
    for path in feat_paths:
        with h5py.File(path, 'r') as f:
            coords = f['coords'][()]

        coord_path = coord_paths + '/' + path.split('/')[-1].replace('.h5', '.pt')
        torch.save(coords, coord_path)

if __name__ == '__main__':
    feat_path = './feat_uni/h5_files'
    coord_paths = './feat_uni/coords'
    save_coord(feat_path, coord_paths)    
