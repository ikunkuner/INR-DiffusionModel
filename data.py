import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tifffile as tif
import os,re
from pathlib import Path
import numpy as np
import random
from torchvision import transforms as T
import matplotlib.pyplot as plt
from typing import Generator, Tuple, List, Union
# --- Helper function for sorting TIFF files by filename ---
def extract_num(file_path: Path) -> int:
    nums = re.findall(r'\d+', file_path.name)
    return int(nums[0]) if nums else 0

def create_grid_3d(z, h, w):
    grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, z), 
                                    torch.linspace(0, 1, h), 
                                    torch.linspace(0, 1, w), indexing='ij'), -1)
    # grid shape is (z, h, w, 3)  
    return grid

class CoordTifDataset(Dataset):
    def __init__(self, volume_path: Path | str):
        """
        Args:
            folder_path: folder_path contain .tif file
        """
        super().__init__()
        self.coord = torch.stack(torch.meshgrid(torch.linspace(0, 1, 256), 
                                                torch.linspace(0, 1, 256), 
                                                torch.linspace(0, 1, 256), 
                                                indexing='ij'), 
                                                -1).float()
        # print(self.coord.shape)

        paths = sorted([p for p in Path(f'{volume_path}').glob(f'*.tif')],key=extract_num) # (32,256,256)
        self.vol = tif.imread(paths)    #(Z,Y,X)

    def __len__(self):
        return self.vol.shape[1]  # ZX_num 

    def __getitem__(self, index):
        # coord [Z,H,W,3] -> [256,256,256,3]
        # img [Z,H,W] -> [256,256,256]
        # return shape is [256,256,3] , [32,256]
        if random.choice([True,False]):
            return self.coord[:, index, :, :], self.vol[:, index,:]/255.0
        else:
            return self.coord[:, :, index, :], self.vol[:, :, index]/255.0
