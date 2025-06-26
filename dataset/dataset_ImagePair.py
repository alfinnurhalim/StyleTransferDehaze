import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from itertools import product

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, phase, img_size=(256, 256), stage=1, suffix='', augment=False, pairing='mix', return_gt=False):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.img_size = img_size
        self.augment = augment
        self.pairing = pairing
        self.return_gt = return_gt

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

        if pairing == 'direct':
            self.pairs = list(product(range(len(self.hazy_files)), range(len(self.clear_files))))
        else:
            self.pairs = None

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs) if self.pairing == 'direct' else len(self.hazy_files)

    def load_img(self, path):
        img = Image.open(path).convert('RGB')

        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img = img.rotate(angle, expand=True)

        img = img.resize(self.img_size, Image.BICUBIC)
        return self.to_tensor(img)

    def __getitem__(self, idx):
        if self.pairing == 'direct':
            i, j = self.pairs[idx]
        else:
            i = idx
            j = random.randint(0, len(self.clear_files) - 1)

        hazy_path = os.path.join(self.hazy_dir, self.hazy_files[i])
        clear_path = os.path.join(self.clear_dir, self.clear_files[j])

        hazy_img = self.load_img(hazy_path)
        clear_img = self.load_img(clear_path)

        if self.return_gt:
            gt_path = os.path.join(self.clear_dir, self.clear_files[i])
            gt_img = self.load_img(gt_path)
            return hazy_img, clear_img, gt_img

        return hazy_img, clear_img
