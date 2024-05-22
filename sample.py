import torch
from utils.modelsother import UNet
from utils.scheduler import NoiseScheduler
import tqdm

T = 100
img_size = 64
num_images = 1

noise_scheduler = NoiseScheduler(T=T, type='quadratic', initial_beta=1e-4, final_beta=0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = UNet()
state_dict = torch.load('model_weights.pth')
model.load_state_dict(state_dict)
model = model.to(device)

alpha_hat = noise_scheduler.get_schedule()
alpha_hat = alpha_hat.to(device)

x = torch.randn((num_images,3,img_size,img_size), device=device)
model.eval()
for t in tqdm.tqdm(range(T,0,-1)):
    with torch.no_grad():
        z = torch.randn_like(x, device=device) if t>1 else torch.zeros_like(x, device=device)
        alpha = alpha_hat[t]/alpha_hat[t-1]
        x = 1/torch.sqrt(alpha)*(x - (1-alpha)/torch.sqrt(1-alpha_hat[t])*model(x,torch.tensor([t], device=device)))
    
torch.save(x.detach().cpu(), 'generated.png')
    
