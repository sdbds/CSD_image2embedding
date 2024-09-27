import io
import os
from torch.utils.data import Dataset
from PIL import Image
import lance

class CustomDataset(Dataset):
    def __init__(self, image_or_lance_path, transform=None):
        self.ds = lance.dataset(image_or_lance_path)
        self.transform = transform

    def __len__(self):
        return self.ds.count_rows()

    def load_image(self, idx):
        raw_img = self.ds.take([idx], columns=["image"]).to_pydict()
        img = Image.open(io.BytesIO(raw_img["image"][0]))
        if img.mode != "RGB":
           img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    
    def load_path(self, idx):
        filename = self.ds.take([idx], columns=["filename"]).to_pydict()
        return filename["filename"][0]

    def __getitem__(self, idx):
        path = self.load_path(idx)
        img = self.load_image(idx)
        return path, img
