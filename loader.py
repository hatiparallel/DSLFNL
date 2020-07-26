import numpy as np
import torch
import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from typing import Tuple

from PIL import Image

IMG_EXTENSIONS = [
    '.png',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root, is_train):
    datanames_file = os.path.join(root, 'meta', 'train.txt')
    if not is_train:
        datanames_file = os.path.join(root, 'meta', 'test.txt')

    class_id = 0
    images = []

    f = open(datanames_file, 'r')
    line = f.readline().strip()
    previous_class_name = list(line.split('/'))[0]

    classwise_idx = [0]

    while line:
        path = os.path.join(root, 'images', (line + '.jpg'))
        class_name = list(line.split('/'))[0]
        if class_name != previous_class_name:
            class_id += 1
            classwise_idx.append(len(images))
        target = class_id

        images.append((path, target))

        previous_class_name = class_name
        line = f.readline().strip()
    f.close()

    classwise_idx.append(len(images))
    classwise_idx = np.array(classwise_idx)

    return images, classwise_idx

def transform_image(img, random):
    resize = transforms.Resize(256)
    crop = transforms.RandomCrop(224)
    flip = transforms.RandomHorizontalFlip()
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([resize, crop, flip, totensor, normalize])
    if not random:
        transform = transforms.Compose([resize, totensor, normalize])
    return transform(img)


class Food101(data.Dataset):

    def __init__(self, is_train, random = True):
        root = '../../../srv/datasets/FoodLog/food-101/'

        imgs, classwise_idx = make_dataset(root, is_train)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 images in subfolders of: ' + root + '\n'
                               'Supported image extensions are: ' + ','.join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.random = random
        self.is_train = is_train
        self.classwise_idx = classwise_idx

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index

        Returns:
            tuple: (input_tensor, target_tensor)
        """
        path, target = self.imgs[idx]
        target = torch.tensor(target, requires_grad = False)
        input_tensor = transform_image(Image.open(path).convert('RGB'), random = self.random)
        return input_tensor, target

    def __len__(self):
        return len(self.imgs)

class Food101OneClass(Food101):
    def __init__(self, class_idx, is_train, random = True):
        imgs = make_dataset_one_class(root, class_idx, is_train)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 images in subfolders of: ' + root + '\n'
                               'Supported image extensions are: ' + ','.join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.random = random
        self.is_train = is_train
