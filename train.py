import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.scheduler import *
import matplotlib.pyplot as plt
from datasets import load_dataset
from utils.modelsother import *
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('./runs/new')
T = 256
img_size = 64

# Scheduler
noise_scheduler = NoiseScheduler(T=T, type='quadratic', initial_beta=1e-4, final_beta=0)

# Dataset
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


# Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = UNet().to(device)


# Training loop
loss_fcn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(epoch_id):
    epoch_loss = 0
    for i, inputs in enumerate(train_dataloader):
        # Take a batch of images
        imgs = inputs['images'].to(device)
        optimizer.zero_grad()
        
        # For each image sample a different time moment
        t = torch.randint(low=1,high=T+1, size=(imgs.shape[0],), device=device) # batch size
        
        # Noise each image in batch according to its time
        noise = torch.randn_like(imgs, device=device)
        alpha_hats = noise_scheduler.get_schedule().to(device)[t]
        imgs = imgs*alpha_hats[:,None,None,None] + noise*torch.sqrt(1-alpha_hats[:,None,None,None])
        
        # Predict the noise
        noise_pred = model(imgs, t)
        
        # Compute loss and backpropagate
        loss = loss_fcn(noise, noise_pred)
        loss.backward()
        
        # Update model weights
        optimizer.step()
        
        # Report loss
        epoch_loss += loss.item()
        
        print(f'Loss epoch[{epoch_id}], step[{epoch_id*len(train_dataloader) + i}]: {loss.item()}')
        writer.add_scalar('Training Loss', loss.item(), epoch_id*len(train_dataloader) + i)
        
    return epoch_loss/len(train_dataloader)

# Train procedure:
num_epochs = 1

loss = []
for i in range(num_epochs):
    model.train()
    epoch_loss = train_epoch(i)
    
    loss.append(epoch_loss)
    
torch.save(model.state_dict(), './model_weights.pth')