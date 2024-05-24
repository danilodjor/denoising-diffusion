import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt

from utils.models import UNet
from utils.scheduler import NoiseScheduler


writer = SummaryWriter("./runs/new")
num_denoising_steps = 256
img_size = 64
num_epochs = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Scheduler
noise_scheduler = NoiseScheduler(
    T=num_denoising_steps, type="quadratic", initial_beta=1e-4, final_beta=0
)
alpha_hats = noise_scheduler.get_schedule().to(device)

# Dataset
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

preprocess = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


# Model
model = UNet().to(device)


# Training loop
loss_fcn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Train procedure:
model.train()
for epoch in range(num_epochs):
    for i, inputs in enumerate(train_dataloader):
        # Take a batch of images
        imgs = inputs["images"].to(device)
        optimizer.zero_grad()

        # For each image sample a different time moment
        t = torch.randint(
            low=1, high=num_denoising_steps + 1, size=(imgs.shape[0],), device=device
        )  # batch size

        # Noise each image in batch according to its time
        alpha_hat = alpha_hats[t]
        noise = torch.randn_like(imgs, device=device)
        imgs = imgs * alpha_hat[:, None, None, None] + noise * torch.sqrt(
            1 - alpha_hat[:, None, None, None]
        )

        # Predict the noise
        noise_pred = model(imgs, t)

        # Compute loss and backpropagate
        loss = loss_fcn(noise, noise_pred)
        loss.backward()

        # Update model weights
        optimizer.step()

        # Report loss
        print(
            f"Epoch {epoch}/{num_epochs}, step {epoch*len(train_dataloader) + i}: {loss.item():.4f}"
        )
        writer.add_scalar(
            "Training Loss", loss.item(), epoch * len(train_dataloader) + i
        )

torch.save(model.state_dict(), "./model_weights.pth")
