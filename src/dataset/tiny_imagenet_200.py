# This file is based on
# https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/tiny_imagenet.py
# (MIT License). Modifications by Kitsuya Azuma, 2024/10/08.
import os
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, pil_loader
from torchvision.datasets.utils import (
    extract_archive,
    check_integrity,
    download_url,
    verify_str_arg,
)


class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
           and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
           target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
           puts it in root directory. If dataset is already downloaded, it is not
           downloaded again.
    """

    base_folder = "tiny-imagenet-200/"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        super(TinyImageNet, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(
            split,
            "split",
            (
                "train",
                "val",
            ),
        )

        if self._check_integrity():
            print("Files already downloaded and verified.")
        elif download:
            self._download()
        else:
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )
        if not os.path.isdir(self.dataset_path):
            print("Extracting...")
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, "wnids.txt"))

        dataset = make_dataset(self.root, self.base_folder, self.split, class_to_idx)
        self.data = [np.array(pil_loader(d[0])) for d in dataset]
        self.targets = [d[1] for d in dataset]

    def _download(self):
        print("Downloading...")
        download_url(self.url, root=self.root, filename=self.filename)
        print("Extracting...")
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def find_classes(class_file):
    with open(class_file) as r:
        classes = [s.strip() for s in r.readlines()]

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == "train":
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, "images")
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, "images")
        imgs_annotations = os.path.join(dir_path, "val_annotations.txt")

        with open(imgs_annotations) as r:
            data_info = [s.split("\t") for s in r.readlines()]

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images
