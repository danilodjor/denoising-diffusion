import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.models import UNet
from utils.scheduler import NoiseScheduler
from utils.config import load_config
from utils.data import get_data


# Config
config = load_config("config.yaml")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

# Scheduler
noise_scheduler = NoiseScheduler(
    T=config["diffusion"]["num_steps"],
    type="quadratic",
    initial_beta=1e-4,
    final_beta=0,
)
alpha_hats = noise_scheduler.get_schedule().to(device)


# Dataset
dataloader = get_data(config)


# Training loop
def train(config):
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    logdir = config["logging"]["log_dir"]
    logdir = os.path.join(logdir, timestamp)
    savedir = config["logging"]["save_dir"]
    savedir = os.path.join(savedir, timestamp)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    num_denoising_steps = config["diffusion"]["num_steps"]

    # Logging
    writer = SummaryWriter(logdir)

    # Model
    model = UNet().to(device)

    # Loss and Optimizer
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train procedure:
    model.train()
    for epoch in range(num_epochs):
        for i, inputs in enumerate(dataloader):
            # Take a batch of images
            imgs = inputs["images"].to(device)
            optimizer.zero_grad()

            # For each image sample a different time moment
            t = torch.randint(
                low=1,
                high=num_denoising_steps + 1,
                size=(imgs.shape[0],),
                device=device,
            )

            # Noise each image in batch according to its time
            alpha_hat = alpha_hats[t]
            noise = torch.randn_like(imgs, device=device)
            imgs = imgs * alpha_hat[:, None, None, None] + noise * torch.sqrt(
                1 - alpha_hat[:, None, None, None]
            )

            # Predict the noise
            noise_pred = model(imgs, t)

            # Compute loss, backpropagate and update weights
            loss = loss_fcn(noise, noise_pred)
            loss.backward()
            optimizer.step()

            # Report loss
            print(
                f"Epoch {epoch}/{num_epochs}, step {epoch*len(dataloader) + i}: {loss.item():.4f}"
            )
            writer.add_scalar("Training Loss", loss.item(), epoch * len(dataloader) + i)

        model_save_path = os.path.join(savedir, f"checkpoint_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)


def main(config):
    train(config)


if __name__ == "__main__":
    main(config)
