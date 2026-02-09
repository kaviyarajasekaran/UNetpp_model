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

fixed_noisy_list = []
fixed_clean_list = []
for noisy, clean in val_loader:
    fixed_noisy_list.append(noisy)
    fixed_clean_list.append(clean)
    if len(fixed_noisy_list) == 3:
        break

fixed_noisy = torch.cat(fixed_noisy_list, dim=0).to(DEVICE)
fixed_clean = torch.cat(fixed_clean_list, dim=0).to(DEVICE)

train_losses = []
val_losses = []

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))


def save_triplet(noisy, clean, pred, save_path):
    noisy = noisy.detach().cpu().permute(1,2,0)
    clean = clean.detach().cpu().permute(1,2,0)
    pred  = pred.detach().cpu().permute(1,2,0)

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    axes[0].imshow(noisy, cmap="gray"); axes[0].set_title("Noisy"); axes[0].axis("off")
    axes[1].imshow(clean, cmap="gray"); axes[1].set_title("Clean"); axes[1].axis("off")
    axes[2].imshow(pred, cmap="gray");  axes[2].set_title("Prediction"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

for epoch in range(Config.epochs):

    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Config.epochs}]")

    for noisy, clean in pbar:
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            pred = model(noisy)
            loss = criterion(pred, clean)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
        for noisy, clean in val_loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            val_loss += criterion(model(noisy), clean).item()

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), f"{Config.checkpoint_dir}/epoch_{epoch+1:02d}.pth")

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
        fixed_pred = model(fixed_noisy)
        fixed_pred = torch.clamp(fixed_pred, 0, 1)

    epoch_dir = f"{Config.prediction_dir}/epoch_{epoch+1:02d}"
    os.makedirs(epoch_dir, exist_ok=True)

    for i in range(3):
        save_triplet(
            fixed_noisy[i],
            fixed_clean[i],
            fixed_pred[i],
            f"{epoch_dir}/sample_{i+1}.png"
        )

    scheduler.step()
