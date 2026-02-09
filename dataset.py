import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_geometric_aug(image_size=384, split="train"):
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                p=0.5
            ),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
        ])


def get_noisy_only_aug():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.15),
    ])


def get_to_tensor():
    return A.Compose([ToTensorV2()])

class DenoisingDataset(Dataset):
    def __init__(
        self,
        noisy_dir,
        clean_dir,
        split="train",
        val_ratio=0.2,
        image_size=384,
        invert_target=False,
        channels=1 
    ):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.split = split.lower()
        self.invert_target = invert_target
        self.channels = channels  # 1 or 3

        noisy_images = sorted(os.listdir(noisy_dir))
        clean_images = sorted(os.listdir(clean_dir))

        noisy_map = {os.path.splitext(f)[0]: f for f in noisy_images}
        clean_map = {os.path.splitext(f)[0]: f for f in clean_images}
        common_keys = sorted(list(set(noisy_map.keys()) & set(clean_map.keys())))

        if len(common_keys) == 0:
            raise ValueError("No matching noisy-clean pairs found!")

        self.noisy_images = [noisy_map[k] for k in common_keys]
        self.clean_images = [clean_map[k] for k in common_keys]

        print(f"Total matched images: {len(self.noisy_images)}")

        indices = list(range(len(self.noisy_images)))
        train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42, shuffle=True)
        self.indices = train_idx if self.split == "train" else val_idx

        self.geo_aug = get_geometric_aug(image_size=image_size, split=self.split)
        self.noisy_aug = get_noisy_only_aug() if self.split == "train" else None
        self.to_tensor = get_to_tensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[real_idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[real_idx])

        if self.channels == 3:
            noisy = np.array(Image.open(noisy_path).convert("RGB"), dtype=np.float32) / 255.0
            clean = np.array(Image.open(clean_path).convert("RGB"), dtype=np.float32) / 255.0
        else:  # channels = 1
            noisy = np.array(Image.open(noisy_path).convert("L"), dtype=np.float32) / 255.0
            clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0

        if self.invert_target:
            clean = 1.0 - clean

        out = self.geo_aug(image=noisy, mask=clean)
        noisy, clean = out["image"], out["mask"]

        if self.noisy_aug is not None:
            noisy = self.noisy_aug(image=noisy)["image"]

        noisy_t = self.to_tensor(image=noisy)["image"].float()
        clean_t = self.to_tensor(image=clean)["image"].float()

        return noisy_t, clean_t
