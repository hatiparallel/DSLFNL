import numpy as np
import torch
import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from PIL import Image

IMG_EXTENSIONS = [
    '.png',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root, is_train):
    datanames_file = os.path.join(root, "meta", "train.txt")
    if not is_train:
        datanames_file = os.path.join(root, "meta", "test.txt")

    class_id = 0
    images = []

    f = open(datanames_file, 'r')
    line = f.readline().strip()
    previous_class_name = list(line.split('/'))[0]

    while line:
        path = os.path.join(root, images, (line + ".jpg"))
        class_name = list(line.split('/'))[0]
        if class_name != previous_class_name:
            class_id += 1
        target = class_id

        images.append((path, target))

        previous_class_name = class_name
        line = f.readline().strip()
    f.close()

    return images


def transform_image(img, random):
    resize = transforms.Resize((64, 64))
    crop = transforms.RandomCrop(64, padding=4)
    flip = transforms.RandomHorizontalFlip()
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([resize, crop, flip, totensor, normalize])
    if not random:
        transform = transforms.Compose([resize, totensor, normalize])
    return transform(img)


class food101(data.Dataset):

    def __init__(self, is_train, random = True):
        root = "../../../srv/datasets/FoodLog/food-101/"

        imgs = make_dataset(root, is_train)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.random = random
        self.is_train = is_train

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor)
        """
        path, target = self.imgs[idx]
        target = torch.tensor([target], requires_grad = False)
        input_tensor = transform_image(Image.open(path).convert("RGB"), random = self.random)
        return input_tensor, target, idx, r_fix

    def __len__(self):
        return len(self.imgs)
