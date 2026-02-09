import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_unetpp import UNetPP
from dataset import DrawingDataset
from loss import L1_Edge_Loss
from torch.utils.data import DataLoader
from config import Config  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.prediction_dir, exist_ok=True)

train_dataset = DrawingDataset(Config.noisy_dir, Config.clean_dir, Config.img_size)
val_dataset   = DrawingDataset(Config.noisy_dir, Config.clean_dir, Config.img_size)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

model = UNetPP().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

criterion = L1_Edge_Loss(alpha=1.0, beta=0.5)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))

fixed_noisy_list, fixed_clean_list = [], []

for noisy, clean in val_loader:
    fixed_noisy_list.append(noisy)
    fixed_clean_list.append(clean)
    if len(fixed_noisy_list) == 3:
        break

fixed_noisy = torch.cat(fixed_noisy_list).to(DEVICE)
fixed_clean = torch.cat(fixed_clean_list).to(DEVICE)

def save_triplet(noisy, clean, pred, save_path):
    noisy = noisy.cpu().permute(1,2,0)
    clean = clean.cpu().permute(1,2,0)
    pred  = pred.cpu().permute(1,2,0)

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(noisy, cmap="gray"); ax[0].set_title("Noisy")
    ax[1].imshow(clean, cmap="gray"); ax[1].set_title("Clean")
    ax[2].imshow(pred, cmap="gray");  ax[2].set_title("Pred")
    for a in ax: a.axis("off")

    plt.savefig(save_path, dpi=200)
    plt.close()

for epoch in range(Config.epochs):
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")

    for noisy, clean in pbar:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            pred = model(noisy)
            loss = criterion(pred, clean)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            val_loss += criterion(model(noisy), clean).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    torch.save(model.state_dict(), f"{Config.checkpoint_dir}/epoch_{epoch+1}.pth")

    with torch.no_grad():
        pred = torch.clamp(model(fixed_noisy), 0, 1)

    epoch_dir = f"{Config.prediction_dir}/epoch_{epoch+1}"
    os.makedirs(epoch_dir, exist_ok=True)

    for i in range(3):
        save_triplet(fixed_noisy[i], fixed_clean[i], pred[i],
                     f"{epoch_dir}/sample_{i}.png")

    scheduler.step()
