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
        elif pairing == 'paired':
            if len(self.hazy_files) != len(self.clear_files):
                raise ValueError(
                    f"Paired mode requires equal file counts, got "
                    f"{len(self.hazy_files)} hazy and {len(self.clear_files)} clear images."
                )
            self.pairs = list(zip(range(len(self.hazy_files)), range(len(self.clear_files))))
        else:
            self.pairs = None

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs) if self.pairing in ['direct', 'paired'] else len(self.hazy_files)

    def get_aug_params(self):
        return {
            'flip': random.random() > 0.5,
            'angle': random.choice([0, 90, 180, 270]),
        }

    def load_img(self, path, aug_params=None):
        img = Image.open(path).convert('RGB')

        if self.augment:
            if aug_params is None:
                aug_params = self.get_aug_params()

            if aug_params['flip']:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            angle = aug_params['angle']
            if angle != 0:
                img = img.rotate(angle, expand=True)

        img = img.resize(self.img_size, Image.BICUBIC)
        return self.to_tensor(img)

    def __getitem__(self, idx):
        if self.pairing in ['direct', 'paired']:
            i, j = self.pairs[idx]
        else:
            i = idx
            j = random.randint(0, len(self.clear_files) - 1)

        hazy_path = os.path.join(self.hazy_dir, self.hazy_files[i])
        clear_path = os.path.join(self.clear_dir, self.clear_files[j])

        aug_params = self.get_aug_params() if self.augment and self.pairing == 'paired' else None
        hazy_img = self.load_img(hazy_path, aug_params)
        clear_img = self.load_img(clear_path, aug_params)

        if self.return_gt:
            gt_path = os.path.join(self.clear_dir, self.clear_files[i])
            gt_img = self.load_img(gt_path, aug_params)
            return hazy_img, clear_img, gt_img

        return hazy_img, clear_img
