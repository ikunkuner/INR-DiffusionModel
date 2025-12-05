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
# --- Helper function for sorting TIFF files by filename ---
def extract_num(file_path: Path) -> int:
    nums = re.findall(r'\d+', file_path.name)
    return int(nums[0]) if nums else 0

# -----------------------------
# ZX  Dateset 
# -----------------------------
class CoordTifDataset(Dataset):
    def __init__(self, volume_path: Path | str):
        """
        Args:
            folder_path: folder_path contain .tif file
        """
        super().__init__()
        X_range = np.linspace(0, 1, 256, endpoint=False)
        Z_range = np.linspace(0, 1, 256, endpoint=False)
        coords = np.stack(np.meshgrid(Z_range, X_range, X_range, indexing='ij'), -1)
        self.coord = torch.from_numpy(coords).float()  #(Z,H,W,3)

        paths = sorted([p for p in Path(f'{volume_path}').glob(f'*.tif')],key=extract_num) # (32,256,256)
        self.vol = tif.imread(paths)    #(Z,Y,X)


    def __len__(self):
        return self.vol.shape[1]  # ZX_num 

    def __getitem__(self, index):
        coord = self.coord[:,index,:]  
        img = (self.vol[:,index,:] / 255.0)   # Y=idx, (Z,X) (32,256)
        
        return coord,img


def main():
    my_data_train = r"/mnt/mira_datacenter/lyh/CodeProjects/diffusion/2Ddiffusion/data/EPFL/task0/test"
    #target_vol = np.random.randint(0,255,size=(256,256,256))
    train_data = CoordTifDataset(Path(my_data_train))
    train_loader = DataLoader(train_data, batch_size = 16, shuffle = False, num_workers = 0)

    """
    print(f"训练集大小: {len(train_loader.dataset)}")
    
    # 显示第一个batch
    print("\n显示训练集的第一个batch:")
    for batch_idx, batch_images in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  形状: {batch_images[0].shape}")
        print(f"  形状: {batch_images[1].shape}")
        print(f"  形状: {batch_images[1].dtype}")
        #print(f"  值域: [{batch_images.min():.4f}, {batch_images.max():.4f}]")
        if batch_idx == 0:  # 仅显示第一个batch
            break
    """
    for batch_idx, batch in enumerate(train_loader): 
        batch_size = batch[1].shape[0] # (B,Z,X)

        fig, axes = plt.subplots(
            nrows=4, ncols=4,  # 4行4列子图（适配batch_size=16）
            figsize=(20, 20),  # 画布大小（宽20英寸，高20英寸，可调整）
            tight_layout=True  # 自动调整子图间距，避免重叠
        )
        
        # 遍历每个子图，显示对应预测结果 
        for idx, ax in enumerate(axes.flatten()):  # axes.flatten() 转为1D迭代器
            if idx < batch_size:  # 防止batch不足16张时索引越界
                # 显示预测图像（*255 转为0-255范围，cmap='gray'灰度显示）
                im = ax.imshow(batch[1][idx], cmap='gray', vmin=0, vmax=255)
                # 添加子图标题（显示第几张图）
                ax.set_title(f'Prediction #{idx+1}', fontsize=14, pad=10)
                # 关闭坐标轴（更干净）
                ax.axis('off')
        
        # 保存画布（可选，高分辨率）
        #plt.savefig(f'./output/CheckExp/ZY_{batch_idx}.png',dpi=150,bbox_inches='tight', pad_inches=0.5)
        
        # 显示画布
        plt.show()
if __name__ == "__main__":
    main()