import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DrawingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, split="train", val_ratio=0.1, img_size=384):

        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        self.files = sorted(os.listdir(noisy_dir))
        print("Total images:", len(self.files))

        if split == "train":
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

        if split == "train":
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy = Image.open(os.path.join(self.noisy_dir, self.files[idx])).convert("L")
        clean = Image.open(os.path.join(self.clean_dir, self.files[idx])).convert("L")

        noisy = np.array(noisy)
        clean = np.array(clean)

        aug = self.transform(image=noisy, mask=clean)
        return aug["image"], aug["mask"]
