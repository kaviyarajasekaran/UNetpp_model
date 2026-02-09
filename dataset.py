import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DrawingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, split="train", val_ratio=0.2, img_size=384):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.split = split

        self.files = sorted(os.listdir(noisy_dir))
        print("Total images:", len(self.files))

        split_idx = int(len(self.files) * (1 - val_ratio))
        self.files = self.files[:split_idx] if split=="train" else self.files[split_idx:]
        print(f"{split} size:", len(self.files))

        if split == "train":
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.2),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy = np.array(Image.open(os.path.join(self.noisy_dir, self.files[idx])).convert("L")) / 255.0
        clean = np.array(Image.open(os.path.join(self.clean_dir, self.files[idx])).convert("L")) / 255.0

        aug = self.transform(image=noisy, mask=clean)
        return aug["image"].float(), aug["mask"].float()
