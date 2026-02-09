import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, split="train", val_ratio=0.2, img_size=384):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.split = split.lower()

        noisy_files = set(os.listdir(noisy_dir))
        clean_files = set(os.listdir(clean_dir))
        self.files = sorted(list(noisy_files.intersection(clean_files)))

        print(f"Total matched images: {len(self.files)}")

        train_files, val_files = train_test_split(
            self.files, test_size=val_ratio, random_state=42
        )
        self.files = train_files if self.split == "train" else val_files
        print(f"{split} size: {len(self.files)}")

        if self.split == "train":
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
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
        fname = self.files[idx]

        noisy_path = os.path.join(self.noisy_dir, fname)
        clean_path = os.path.join(self.clean_dir, fname)

        noisy = np.array(Image.open(noisy_path).convert("L"), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0

        aug = self.transform(image=noisy, mask=clean)
        noisy = aug["image"].float()
        clean = aug["mask"].float()

        return noisy, clean
