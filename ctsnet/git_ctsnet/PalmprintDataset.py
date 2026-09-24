import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class PalmprintDataset(Dataset):
    """
    Palmprint dataset loader for reading image-path-label text file.
    """

    def __init__(self, txt=None, transform=T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ]), target_size=(128, 128), return_path=False):
        """
        Args:
            txt: path of text file containing image path and label
            transform: torchvision transform pipeline for image preprocessing
            target_size: spatial size to resize input images (height, width)
            return_path: whether to return original image path together with data
        """
        self.text_path = txt
        self.transform = transform
        self.target_size = target_size
        self.return_path = return_path

        self.img_paths = []
        self.labels = []

        if txt is not None:
            self._read_txt_file()

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.img_paths)

    def _read_txt_file(self):
        """Read image paths and labels from specified text file."""
        self.img_paths = []
        self.labels = []

        with open(self.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                self.img_paths.append(parts[0])
                self.labels.append(parts[1])

    def get_image_paths(self):
        """Return list of all image paths."""
        return self.img_paths

    def __getitem__(self, idx):
        """
        Load and preprocess single sample by index.
        Args:
            idx: sample index
        Returns:
            image tensor, label tensor, (optional) image path
        """
        img_path = self.img_paths[idx]
        # Read grayscale image
        img = cv2.imread(img_path, 0)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        # Resize image to target resolution
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Convert to PIL Image and apply transform pipeline
        img_pil = Image.fromarray(img)
        if self.transform is not None:
            img_pil = self.transform(img_pil)

        # Convert label to integer tensor
        label = torch.tensor(int(self.labels[idx]))

        if self.return_path:
            return img_pil, label, img_path
        return img_pil, label