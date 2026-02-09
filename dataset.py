from torch.utils.data import Dataset
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, split="train", val_ratio=0.2, img_size=256):

        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        noisy_files = os.listdir(noisy_dir)
        clean_files = os.listdir(clean_dir)

        noisy_names = set([f.split('.')[0] for f in noisy_files])
        clean_names = set([f.split('.')[0] for f in clean_files])

        self.files = list(noisy_names.intersection(clean_names))
        print(f"Total matched images: {len(self.files)}")

        train_files, val_files = train_test_split(self.files, test_size=val_ratio, random_state=42)

        self.files = train_files if split=="train" else val_files

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        noisy_path = os.path.join(self.noisy_dir, name + ".png")
        clean_path = os.path.join(self.clean_dir, name + ".png")

        noisy = Image.open(noisy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        return self.transform(noisy), self.transform(clean)
