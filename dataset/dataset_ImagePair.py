import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, phase, img_size=(256, 256), stage=1, suffix='', augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.img_size = img_size
        self.augment = augment

        self.hazy_dir = os.path.join(root_dir, f"{phase}A" + suffix)
        self.clear_dir = os.path.join(root_dir, f"{phase}B" + suffix)

        self.hazy_files = sorted([
            f for f in os.listdir(self.hazy_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ])

        self.clear_files = sorted([
            f for f in os.listdir(self.clear_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ])

        assert len(self.hazy_files) == len(self.clear_files), "Mismatch in number of images"

    def __len__(self):
        return len(self.hazy_files)

    def paired_transform(self, hazy, clear):
        # Random horizontal flip
        if random.random() > 0.5:
            hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation (0, 90, 180, 270 degrees)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            hazy = hazy.rotate(angle, expand=True)
            clear = clear.rotate(angle, expand=True)

        # Random crop to target size
        # w, h = hazy.size
        # th, tw = self.img_size
        # if w > tw and h > th:
        #     x1 = random.randint(0, w - tw)
        #     y1 = random.randint(0, h - th)
        #     hazy = hazy.crop((x1, y1, x1 + tw, y1 + th))
        #     clear = clear.crop((x1, y1, x1 + tw, y1 + th))
        # else:
        #     # Resize first if too small
        #     hazy = hazy.resize(self.img_size, Image.BICUBIC)
        #     clear = clear.resize(self.img_size, Image.BICUBIC)

        return hazy, clear

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_files[idx])
        clear_path = os.path.join(self.clear_dir, self.clear_files[idx])

        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        if self.augment:
            hazy_img, clear_img = self.paired_transform(hazy_img, clear_img)
        else:
            hazy_img = hazy_img.resize(self.img_size, Image.BICUBIC)
            clear_img = clear_img.resize(self.img_size, Image.BICUBIC)

        transform = transforms.ToTensor()
        return transform(hazy_img), transform(clear_img)
