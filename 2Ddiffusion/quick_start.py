import argparse
import json
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def main(args):
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="config_path")
    args = parser.parse_args()
    main(args)