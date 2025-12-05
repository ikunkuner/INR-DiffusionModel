import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# -----------------------------
# 1. 超参数
# -----------------------------
load_path = r"./ckpt/INR_exp0/inr_epoch_499.pth"
output_path = r"./output/INR_exp1"
os.makedirs(output_path,exist_ok=True)
slice_kind = "ZY" # XY ZX ZY

# ====================== 2. 加载完整模型（推理/继续训练） ======================
# 加载模型（需先定义模型结构 inr_network）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Inr = torch.load(load_path, weights_only=False).to(device)
Inr.eval()

X_range = np.linspace(0, 1, 256, endpoint=False)
Z_range = np.linspace(0, 1, 256, endpoint=False)
coords = np.stack(np.meshgrid(Z_range, X_range, X_range, indexing='ij'), -1) # (Z,Y,X)
coord = torch.from_numpy(coords).float()  #(Z,H,W,3)

if slice_kind == "ZX":
    coord = coord.permute(1,0,2,3)
elif slice_kind == "ZY":
    coord = coord.permute(2,0,1,3)
else: 
    coord = coord # XY

test_loader = DataLoader(coord, batch_size = 16, shuffle = False, num_workers = 0)

for batch_idx, batch in enumerate(test_loader):
    batch_size = batch.shape[0]
    x = batch.reshape(-1, 3).to(device) # (B,Z,X,3)
    with torch.no_grad():  # 关键：禁用梯度，Tensor无需detach
        ypreds = Inr(x, 'sigmoid')   # (B,Z,X)
    #print(f"ypreds.shape:{ypreds.shape}")
    ypreds = (ypreds.reshape(batch_size,256,256).cpu().numpy()*255.0).astype(np.uint8) # (B,Z,X)
    #print(f"ypreds.shape:{ypreds.shape}")
    #print("batch[1].shape:",batch.shape)
    
    # ---------------------- 核心：创建大画布和子图网格 ----------------------
    fig, axes = plt.subplots(
        nrows=4, ncols=4,  # 4行4列子图（适配batch_size=16）
        figsize=(20, 20),  # 画布大小（宽20英寸，高20英寸，可调整）
        tight_layout=True  # 自动调整子图间距，避免重叠
    )
    
    # 遍历每个子图，显示对应预测结果 
    for idx, ax in enumerate(axes.flatten()):  # axes.flatten() 转为1D迭代器
        if idx < batch_size:  # 防止batch不足16张时索引越界
            # 显示预测图像（*255 转为0-255范围，cmap='gray'灰度显示）
            im = ax.imshow(ypreds[idx], cmap='gray', vmin=0, vmax=255)
            # 添加子图标题（显示第几张图）
            ax.set_title(f'Prediction #{idx+1}', fontsize=14, pad=10)
            # 关闭坐标轴（更干净）
            ax.axis('off')
    
    # 保存画布（可选，高分辨率）
    output_file_name = os.path.join(output_path, f"{slice_kind}_{batch_idx}.png")
    plt.savefig(output_file_name,dpi=150,bbox_inches='tight', pad_inches=0.5)
    
    # 显示画布
    #plt.show()
    #plt.imshow(ypreds[0] * 255, cmap='gray')
    #plt.show()