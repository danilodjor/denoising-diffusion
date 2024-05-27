import os
import tqdm
from PIL import Image, ImageDraw
from datetime import datetime

import torch
import torchvision

from utils.models import UNet
from utils.config import load_config
from utils.scheduler import NoiseScheduler


config = load_config("config.yaml")


def sample(config):
    # Config
    num_denoising_steps = config["diffusion"]["num_steps"]
    img_size = config["data"]["img_size"]
    save_dir = config["sampling"]["save_dir"]
    model_path = config["sampling"]["model_path"]
    scheduler_type = config["diffusion"]["scheduler"]
    
    nrow, ncol = config["sampling"]["num_sample_imgs"].split('x')
    nrow, ncol = int(nrow), int(ncol)
    num_images = nrow*ncol

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Sampling loop
    noise_scheduler = NoiseScheduler(
        T=num_denoising_steps, type=scheduler_type, initial_beta=1e-4, final_beta=0.02
    )

    model = UNet()
    if model_path is not None:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    else:
        print("No model to load.")
    model = model.to(device)
    model.eval()

    alpha_hat = noise_scheduler.get_schedule().to(device)
    beta = noise_scheduler.get_beta().to(device)
    alphas = noise_scheduler.get_alpha().to(device)

    x = torch.randn((num_images, 3, img_size, img_size), device=device)
    pbar = tqdm.tqdm(range(num_denoising_steps-1, -1, -1))
    for t in pbar:
        with torch.no_grad():
            time = torch.tensor([t] * num_images, device=device)
            if t > 0:
                z = torch.randn_like(x, device=device)
            else:
                z = torch.zeros_like(x, device=device)

            alpha = alphas[t].unsqueeze(-1)[:, None, None, None]
            sigma = torch.sqrt(beta[t]).unsqueeze(-1)[:, None, None, None]

            noise_pred = model(x, time)

            x = 1/torch.sqrt(alpha) * (x - (1-alpha)/torch.sqrt(1 - alpha_hat[t]) * noise_pred) + sigma*z

    x = (x.clamp(-1, 1) + 1) / 2
    x = torchvision.utils.make_grid(x, nrow=nrow)
    x = (x * 255).type(torch.uint8).permute(1, 2, 0).detach().cpu().numpy()
    
    im = Image.fromarray(x)
    
    I1 = ImageDraw.Draw(im)
    I1.text((10, 10), f"dataset:{config["data"]["dataset"]},\n size={img_size},\n model_path={model_path}", fill=(255, 0, 0), font_size=10)
    
    img_path = os.path.join(save_dir, f"generated_{timestamp}.png")
    im.save(img_path)


def main():
    sample(config)


if __name__ == "__main__":
    main()
