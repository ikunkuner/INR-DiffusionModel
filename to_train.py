import os

os.environ["WANDB_MODE"] = "offline"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.utils.data import DataLoader
import denoising_diffusion_pytorch as ddp
import wandb
import time
import argparse
from pathlib import Path


from INR.Siren_FF import Siren
from data import CoordTifDataset


   # output.log 的路


def main(args):
    # load model
    config_path = args.config_path

    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"程序可见的显卡数量：{torch.cuda.device_count()}")
    print(f"当前使用的逻辑显卡ID：{torch.cuda.current_device()}")
    print(f"当前显卡逻辑ID对应的物理显卡型号：{torch.cuda.get_device_name(0)}")

    unet = Unet(
        dim = config["unet"]["dim"],
        dim_mults = tuple(config["unet"]["dim_mults"]),
        channels = config["unet"]["channels"]
    ).to(device)

    print(f'Model parameter count: {sum(p.numel() for p in unet.parameters()):,}')

    diffusion = GaussianDiffusion(
        unet,
        image_size = config["diffusion"]["image_size"],
        timesteps = config["diffusion"]["timesteps"],
        sampling_timesteps = config["diffusion"]["sampling_timesteps"]
    ).to(device)

    trainer = Trainer(
        diffusion,
        folder = config["trainer"]["data_path"],
        train_batch_size = config["trainer"]["train_batch_size"],
        gradient_accumulate_every = config["trainer"]["gradient_accumulate_every"],
        train_lr=config["trainer"]["train_lr"],
        train_num_steps=config["trainer"]["train_num_steps"],
        ema_decay=config["trainer"]["ema_decay"],
        save_and_sample_every=config["trainer"]["save_and_sample_every"],
        num_samples=config["trainer"]["num_samples"],
        results_folder=config["trainer"]["results_folder"],
        mixed_precision_type=config["trainer"]["mixed_precision_type"],
        calculate_fid=config["trainer"]["calculate_fid"]
    )
    # train 
    train_load_num = config["trainer"]["train_num_steps"] // config["trainer"]["save_and_sample_every"]
    trainer.load(train_load_num)
    trainer.model.eval()

    print("pre-trained diffusion model")
    """
    exp0 = diffusion.sample(batch_size=4).to(device)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True)

    for idx, img_tensor in enumerate(exp0):
        ax = axes[idx // 2, idx % 2]
        
        # NCHW → HWC
        np_array = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
        
        # single-channel squeeze
        if np_array.shape[-1] == 1:
            np_array = np_array.squeeze(-1)
            ax.imshow(np_array, cmap="gray")
        else:
            ax.imshow(np_array)

        ax.set_title(f"Sample {idx+1}")
        ax.axis("off")

    plt.show()
    """

    ## SDS (Score Distillation Sampling) Loss详解

    data_path =args.data_path

    # 定义保存路径（建议按epoch命名，方便追溯）
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在


    inr_network = Siren().to(device)
    optimizer = torch.optim.Adam(inr_network.parameters(), lr=1e-5)

    print("模型所在设备:", next(inr_network.parameters()).device)  # 应该输出 cuda:0

    epochs = 500

    loss_mse = nn.MSELoss(reduction="mean")

    # -----------------------------
    # DataLoader
    # -----------------------------
    train_data = CoordTifDataset(data_path)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)

    # -----------------------------
    # 初始化 WandB
    # -----------------------------
    wdb = wandb.init(
        project= args.project_name,
        name= args.exp_name,
        notes= args.exp_note,
        config={
            "epochs": epochs,
            "lr": 1e-5,
            "batch_size": 4,
            "model": "SIREN+SDS",
            "diffusion_model_path": config_path,
            "diffusion_model_config": config,
            "train_load_num":train_load_num,
            "data_path": data_path,
            "save_dir": save_dir
        },
        settings=wandb.Settings(console="off")
    )

    log_file = Path(wandb.run.dir) / "output.log"   # output.log 的路径

    def write_log(msg):
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    print("WandB logging enabled.")

    # -----------------------------
    # Train start
    # -----------------------------
    for epoch in tqdm(range(epochs), desc="Training epochs", leave=False):

        epoch_loss_data = 0.0
        epoch_loss_sds = 0.0
        epoch_loss_total = 0.0
        
        start_time = time.time()

        # epoch级进度条
        epoch_bar = tqdm(enumerate(train_loader),total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, batch in epoch_bar:

            batch_size, Z, X, C = (*batch[1].shape, 1)  # (B,Z,X,1)

            coord = batch[0].reshape(-1, 3).to(device)
            y_pred = inr_network(coord, 'sigmoid')   # (B,Z,X,C,1) (B,256,256,1,1)
            y_pred = y_pred.reshape(batch_size, 1, 256, 256)

            # -----------------------------
            # 1. Data Loss
            # -----------------------------
            loss_data = loss_mse(y_pred[:,:,::8], batch[1].to(device).float().unsqueeze(1))

            # -----------------------------
            # 2. Diffusion SDS Loss
            # -----------------------------
            with torch.no_grad():
                timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
                noise = torch.randn_like(y_pred)

                noisy_images = diffusion.q_sample(
                    x_start=y_pred, 
                    t=timesteps, 
                    noise=noise
                )

                noise_pred = diffusion.model_predictions(noisy_images, timesteps).pred_noise

                betas = ddp.sigmoid_beta_schedule(1000).to(device)
                beta_t = betas[timesteps]

                sds_grad = beta_t[:, None, None, None] * (noise_pred - noise)

                target = (y_pred - sds_grad).float()

            sds_loss = 0.5 * loss_mse(y_pred, target)

            # -----------------------------
            # 3. Total Loss
            # -----------------------------
            total_loss = loss_data + 0.25 * sds_loss

            # -----------------------------
            # 4. Backprop
            # -----------------------------
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss_data += loss_data.item()
            epoch_loss_sds += sds_loss.item()
            epoch_loss_total += total_loss.item()

            # -----------------------------
            # 5. 更新 batch 进度条信息
            # -----------------------------
            epoch_bar.set_postfix({
                "data_loss": f"{loss_data.item():.6f}",
                "sds_loss": f"{sds_loss.item():.6f}",
                "total": f"{total_loss.item():.6f}"
            })
            wdb.log({
                "global step": epoch * len(train_loader) + batch_idx,
                "batch/data_loss": loss_data.item(),
                "batch/sds_loss": sds_loss.item(),
                "batch/total_loss": total_loss.item(),
                "batch/gpu_mem_MB": torch.cuda.memory_allocated() / 1024 / 1024,
                "batch/lr": optimizer.param_groups[0]["lr"]
            })
            if batch_idx % 16 ==0:
                write_log(
                    f"[Epoch {epoch} | Batch {batch_idx}/{len(train_loader)}] "
                    f"data_loss={loss_data.item():.6f}, "
                    f"sds_loss={sds_loss.item():.6f}, "
                    f"total={total_loss.item():.6f}"
                )

            #epoch_bar.update(1)

        # 一个 epoch 结束
        epoch_time = time.time() - start_time
        wdb.log({
            "epoch": epoch,
            "epoch/data_loss": epoch_loss_data / len(train_loader),
            "epoch/sds_loss": epoch_loss_sds / len(train_loader),
            "epoch/total_loss": epoch_loss_total / len(train_loader),
            "epoch/gpu_mem_MB": torch.cuda.memory_allocated() / 1024 / 1024,
            "epoch/time_sec": epoch_time
        })
        write_log(
            f"[Epoch {epoch} DONE] "
            f"avg_data_loss={epoch_loss_data/len(train_loader):.6f}, "
            f"avg_sds_loss={epoch_loss_sds/len(train_loader):.6f}, "
            f"avg_total_loss={epoch_loss_total/len(train_loader):.6f}, "
            f"time={epoch_time:.2f}s"
        )
        
        # 保存模型
        if (epoch % 100 == 0) or (epoch == epochs -1):
            save_path = os.path.join(save_dir, f"inr_epoch_{epoch}.pth")
            torch.save(inr_network, save_path)
            wandb.save(save_path)

        epoch_bar.close()
    
    # Finish the run and upload any remaining data.
    wdb.finish()


    # ====================== 2. 加载完整模型（推理/继续训练） ======================
    # 加载模型（需先定义模型结构 inr_network）

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="config_path")
    parser.add_argument('--data_path', type=str, required=True, help="data_path")
    parser.add_argument('--save_dir', type=str, required=True, help="data_path")
    # wandb
    parser.add_argument("--project_name", type=str, default="INR-with-Diffusion", help="project_name")
    parser.add_argument("--exp_name", type=str, required=True, help="exp_name")
    parser.add_argument("--exp_note", type=str, required=True, help="exp_note")

    args = parser.parse_args() 
    main(args)