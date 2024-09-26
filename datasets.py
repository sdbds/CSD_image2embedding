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
    
    def load_hash(self, idx):
        hash = self.ds.take([idx], columns=["hash"]).to_pydict()
        return hash["hash"][0]

    def __getitem__(self, idx):
        hash = self.load_hash(idx)
        img = self.load_image(idx)
        return hash, img
