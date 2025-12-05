import argparse
import numpy as np
import os
from tqdm import tqdm
from skimage.restoration import denoise_tv_chambolle
import torch.nn.functional as F
import torch
import tifffile as tiff
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import json
from scipy.ndimage import gaussian_filter

def main(args):
    vol_data = tiff.imread(r"/mnt/mira_datacenter/lyh/MiniDatasets/ElectronMicroscopyDataset/volumedata.tif")
    Z,Y,X = vol_data.shape

    fetch_size = (1024,1024,1024)
    fetch_z = np.random.randint(0,Z-fetch_size[0]+1)
    fetch_y = np.random.randint(0,Y-fetch_size[1]+1)
    fetch_x = np.random.randint(0,X-fetch_size[2]+1)  

    vol_data = vol_data[fetch_z:fetch_z+fetch_size[0], fetch_y:fetch_y+fetch_size[1], fetch_x:fetch_x+fetch_size[2]]
    print("fetch TIF 体数据形状(Z/Y/X ):", vol_data.shape)

    vol_filtered = gaussian_filter(vol_data, sigma=[4.0, 0.5, 0.5]) #[z_start:z_start+z_num, y_start:y_start+block_size[0], x_start:x_start+block_size[1]]
    vol_downsample = vol_filtered[::8,:,:]
    Z,Y,X = vol_downsample.shape
    print("fetch TIF 体数据形状(Z/Y/X ):", vol_downsample.shape)
    
    # === output_path ===
    output_path = "./2Ddiffusion/data/EPFL/task1"
    os.makedirs(output_path, exist_ok=True)
    output_path_train = os.path.join(output_path,"train")
    os.makedirs(output_path_train, exist_ok=True)
    output_path_test = os.path.join(output_path,"test")
    os.makedirs(output_path_test, exist_ok=True)

    train_z_num = Z
    train_patch_size = (256,256)
    train_z_start = 0
    train_y_start = np.random.randint(0,Y-train_patch_size[0]+1)
    train_x_start = np.random.randint(0,X-train_patch_size[1]+1)


    test_z_num = min(32,Z)
    test_patch_size = (256,256)
    test_z_start = np.random.randint(0,Z-test_z_num+1)
    test_y_start = np.random.randint(0,Y-test_patch_size[0]+1)
    test_x_start = np.random.randint(0,X-test_patch_size[1]+1)

    metadata = {
        "fetch_size": fetch_size,
        "fetch_coor":(fetch_z,fetch_y,fetch_x),

        "train_z_num":train_z_num,
        "train_patch_size":train_patch_size,
        "train_z_start":train_z_start,
        "train_y_start":train_y_start,
        "train_x_start":train_x_start,

        "test_z_num":test_z_num,
        "test_patch_size":test_patch_size,
        "test_z_start":test_z_start,
        "test_y_start":test_y_start,
        "test_x_start":test_x_start
    }
    with open(os.path.join(output_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    for z_idx in tqdm(range(train_z_num)):
        tif_path = os.path.join(output_path_train, f"volumedata_{z_idx:04d}.tif")
        tiff.imwrite(tif_path, vol_downsample[train_z_start+z_idx, train_y_start:train_y_start+train_patch_size[0], train_x_start:train_x_start+train_patch_size[1]])
    for z_idx in tqdm(range(test_z_num)):
        tif_path = os.path.join(output_path_test, f"volumedata_{z_idx:04d}.tif")
        tiff.imwrite(tif_path, vol_downsample[test_z_start+z_idx, test_y_start:test_y_start+test_patch_size[0], test_x_start:test_x_start+test_patch_size[1]])
    
# /mnt/mira_datacenter/lyh/CodeProjects/diffusion/2Ddiffusion/results/EPFL/Exp1/config.json  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', choices=['tain', 'test'], required=True, help="")

    args = parser.parse_args()
    main(args)