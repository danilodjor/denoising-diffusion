import os
from PIL import Image
from datetime import datetime
import torch
from utils.models import UNet
from utils.scheduler import NoiseScheduler
import tqdm
from utils.config import load_config


config = load_config("config.yaml")

def sample(config):
    # Config
    num_denoising_steps = config["diffusion"]["num_steps"]
    img_size = config["data"]["img_size"]
    num_images = config["sampling"]["num_sample_imgs"]
    save_dir = config["sampling"]["save_dir"]
    model_path = config["sampling"]["model_path"]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    # Sampling loop
    noise_scheduler = NoiseScheduler(T=num_denoising_steps, type='quadratic', initial_beta=1e-4, final_beta=0)

    model = UNet()
    if model_path is not None:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    else:
        print('No model to load.')
    model = model.to(device)

    alpha_hat = noise_scheduler.get_schedule()
    alpha_hat = alpha_hat.to(device)

    x = torch.randn((num_images,3,img_size,img_size), device=device)
    model.eval()
    for t in tqdm.tqdm(range(num_denoising_steps,0,-1)):
        with torch.no_grad():
            time = torch.tensor([t], device=device)
            z = torch.randn_like(x, device=device) if t>1 else torch.zeros_like(x, device=device)
            alpha = alpha_hat[t]/alpha_hat[t-1]
            noise_pred = model(x, time)
            x = 1/torch.sqrt(alpha)*(x - (1-alpha)/torch.sqrt(1-alpha_hat[t])*noise_pred)
        
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)[0].permute(1, 2, 0).detach().cpu().numpy()
    img_path = os.path.join(save_dir, f"generated_{timestamp}.png")
    
    im = Image.fromarray(x)
    im.save(img_path)
    # torch.save(x.detach().cpu(), img_path)
    

def main():
    sample(config)
    

if __name__ == '__main__':
    main()