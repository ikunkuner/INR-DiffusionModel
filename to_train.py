import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import torch
import argparse
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.utils.data import DataLoader

from INR.Siren_FF import Siren
from data import CoordTifDataset

def main(args):
    """
    Train a single block at a fixed size (32,256,256)
    """
    
    # -----------------------------
    # Load pretrained diffusion model
    # -----------------------------

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unet = Unet(
        dim = config["unet"]["dim"],
        dim_mults = tuple(config["unet"]["dim_mults"]),
        channels = config["unet"]["channels"]
    ).to(device)

    # print(f'Model parameter count: {sum(p.numel() for p in unet.parameters()):,}')

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

    train_load_num = config["trainer"]["train_num_steps"] // config["trainer"]["save_and_sample_every"]
    trainer.load(train_load_num)
    trainer.model.eval()

    # -----------------------------
    # Train INR
    # -----------------------------

    data_path =args.data_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    inr_network = Siren().to(device)
    optimizer = torch.optim.Adam(inr_network.parameters(), lr=1e-5)
    loss_mse = nn.MSELoss(reduction="mean")
    
    train_data = CoordTifDataset(data_path)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)

    epochs = 500

    print("Experiment:")
    print("data_path:", args.data_path)
    print("save_dir:", args.save_dir)
    print("epochs:", epochs)
    for epoch in tqdm(range(1,epochs+1), desc="Training epochs", leave=False):

        epoch_bar = tqdm(enumerate(train_loader),total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch_idx, batch in epoch_bar:

            B, Z, X, C = (*batch[1].shape, 1)  # img (B,32,256,1) B = 256/4 = 64

            coord = batch[0].reshape(-1, 3).to(device) # 
            y_pred = inr_network(coord, 'sigmoid')  # (B,Z,X,C,1)  (B,256,256,1,1)
            y_pred = y_pred.reshape(B, 1, 256, 256) 

            # -----------------------------
            # 1. Data Loss
            # -----------------------------
            
            # plane 1: 
            y_pred_downsample = torch.nn.functional.interpolate(y_pred, size=(32,256), mode='bilinear', antialias=True) # downsample the queried slice to same resolution as measurement, [B, 1, 256, 256] > [B, 1, 26, 256]
            
            # plane 2: 
            # pool = nn.AvgPool2d(kernel_size=(8, 1), stride=(8, 1))  # 或 MaxPool2d # 256 -> 32
            # y_pred_downsample = pool(y_pred)

            # plane 3: 
            # y_pred_downsample = y_pred[:,:,::8,:]

            loss_data = loss_mse(y_pred_downsample, batch[1].to(device).float().unsqueeze(1))

            # -----------------------------
            # 2. Diffusion SDS Loss
            # -----------------------------
            with torch.no_grad(): # (0,1000)
                timesteps = (torch.ones(1)* int(epochs - epoch)).long().to(device) 
                noise = torch.randn_like(y_pred)
                noisy_images = diffusion.q_sample(
                    x_start=y_pred, 
                    t=timesteps, 
                    noise=noise
                )
                noise_pred,*_ = diffusion.model_predictions(noisy_images, timesteps)
                sds_grad = 1 * torch.nan_to_num(noise_pred - noise)

            sds_loss = torch.mul(sds_grad, y_pred).mean()

            # -----------------------------
            # 3. Total Loss
            # -----------------------------
            total_loss = loss_data  + 0.25 * sds_loss

            # -----------------------------
            # 4. Backprop
            # -----------------------------
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # -----------------------------
            # 5. update epoch_bar
            # -----------------------------
            epoch_bar.set_postfix({
                "data_loss": f"{loss_data.item():.6f}",
                "sds_loss": f"{sds_loss.item():.6f}",
                "total": f"{total_loss.item():.6f}"
            })

        # save model
        if (epoch) == epochs:
            save_path = os.path.join(save_dir, f"last_model.pth")
            torch.save(inr_network, save_path)

        epoch_bar.close()
    print("finish!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="config_path")
    parser.add_argument('--data_path', type=str, required=True, help="data_path")
    parser.add_argument('--save_dir', type=str, required=True, help="data_path")
    args = parser.parse_args() 
    main(args)
