import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DrawingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, img_size=384):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.files = sorted(os.listdir(noisy_dir))

        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy = Image.open(os.path.join(self.noisy_dir, self.files[idx]))
        clean = Image.open(os.path.join(self.clean_dir, self.files[idx]))

        noisy = self.transform(noisy)
        clean = self.transform(clean)

        return noisy, clean
