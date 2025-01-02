from collections import Counter
import logging
import os
import re
from typing import List
import torch
from tnibs.data.modules.data import Data
from tnibs.utils._utils import *
import numpy as np
import torch.utils.data as td
from tnibs.data.utils import dfs
from sklearn.preprocessing import LabelEncoder

import glob

from tnibs.utils.array import to_list


class ClassifierData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_attr()
        # Use set_daata

    def decode_label(self, label):
        """Returns name of original label for interpretation. Usually overriden or provisioned by a method below."""
        if self.le:
            try:
                return self.le.inverse_transform(label)
            except ValueError:
                return self.le.inverse_transform([label])[0]
        return label

    def encode_label(self, label):
        """Returns value of encoded label for interpretation. Usually overriden or provisioned by a method below."""
        if self.le:
            try:
                return self.le.transform(label)
            except ValueError:
                return self.le.transform([label])[0]
        return label

    @property
    def classes(self):
        return len(self.le.classes_)

    def set_data(self, *args, test=False, show_counts=True):
        """Initializes self.data

        Args:
            test (bool, optional): Initialize test set. Defaults to False.
            show_counts (bool, optional): Displays data counts on non-test initialization. Defaults to True.
        """
        self.le = LabelEncoder()
        labels = to_list(args[-1])  # counter needs a hashable type

        data = tuple(np.array(a) if isinstance(a, List) else a for a in args[:-1]) + (
            torch.tensor(self.le.fit_transform(labels), dtype=torch.int64),
        )

        if test:
            self.test_data = data
            self.test_dataset = None
        else:
            self.data = data
            self.dataset = None

        # counts

        self.counts = sorted(
            list(Counter(labels).items()), key=lambda x: self.encode_label(x[0])
        )

        if show_counts and test is False:
            self.show_counts()

    def show_counts(self):
        assert all((self.counts, self.data, self.le))
        print("data[-1] counts in label order from 0:")
        for k, v in self.counts:
            print(f"{k}: {v}")

    @classmethod
    def make_inverse_sampler(cls, factor=1):
        """Creates a factory method for creating an weighted random sampler which weights each class according 1/class_size^alpha

        Args:
            factor (int, optional): _description_. Defaults to 1.
        """

        def inner(self, inds, **kwargs):
            nonlocal factor
            assert all((self.data, self.counts))

            def idx_to_label(n):
                return self.data[-1][inds[n]]

            weights = [
                pow(self.counts[idx_to_label(i)][1], -factor) for i in range(len(inds))
            ]

            return td.WeightedRandomSampler(
                weights,
                num_samples=kwargs.pop("num_samples", len(inds)),
                **kwargs,
            )

        return inner


class ImageData(ClassifierData):
    # Use set_data
    def _paths_and_labels_from_folder(
        self,
        folder,
        glob_string,
        label_fn=lambda folder_name: folder_name,
        label_encode=True,
    ):
        paths = []
        labels = []
        for label in os.listdir(folder):
            label_path = os.path.join(folder, label)
            if os.path.isdir(label_path):
                # Get all .jpg files ending with 'Ch3.ome.jpg' in this folder
                image_paths = glob.glob(os.path.join(label_path, glob_string))
                paths.extend(image_paths)
                labels.extend([label_fn(label)] * len(image_paths))
        return paths, labels

    def _paths_and_labels_from_file(self, file, file_label_regex):
        paths = []
        labels = []
        pattern = re.compile(file_label_regex)
        with open(file, "r") as f:
            for line in f.readlines():
                match = pattern.search(line)
                try:
                    paths.append(match.group(1))
                    labels.append(match.group(2))
                except (IndexError, AttributeError):
                    logging.warning(f"Line {line} doesn't match regex")

        return paths, labels

    def visualize(self, imgs, labels=None, grid=(4, 4), title=None):
        """
        input imgs can be single or multiple tensor(s), this function uses matplotlib to visualize.
        Single input example:
        show(x) gives the visualization of x, where x should be a torch.Tensor
            if x is a 4D tensor (like image batch with the size of b(atch)*c(hannel)*h(eight)*w(eight), this function splits x in batch dimension, showing b subplots in total, where each subplot displays first 3 channels (3*h*w) at most.
            if x is a 3D tensor, this function shows first 3 channels at most (in RGB format)
            if x is a 2D tensor, it will be shown as grayscale map

        Multiple input example:
        show(x,y,z) produces three windows, displaying x, y, z respectively, where x,y,z can be in any form described above.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        flag = True
        if isinstance(imgs, (torch.Tensor, np.ndarray)):
            imgs = imgs.detach().cpu()

            if imgs.dim() == 4:  # 4D tensor
                bz = imgs.shape[0]
                c = imgs.shape[1]
                if bz == 1 and c == 1:  # single grayscale image
                    imgs = [imgs.squeeze()]
                elif bz == 1 and c == 3:  # single RGB image
                    imgs = imgs.squeeze()
                    imgs = [imgs.permute(1, 2, 0)]
                elif bz == 1 and c > 3:  # multiple feature maps
                    imgs = imgs[:, 0:3, :, :]
                    imgs = [imgs.permute(0, 2, 3, 1)[:]]
                    print(
                        "warning: more than 3 channels! only channels 0,1,2 are preserved!"
                    )
                elif bz > 1 and c == 1:  # multiple grayscale images
                    imgs = imgs.squeeze()
                elif bz > 1 and c == 3:  # multiple RGB images
                    imgs = imgs.permute(0, 2, 3, 1)
                elif bz > 1 and c > 3:  # multiple feature maps
                    imgs = imgs[:, 0:3, :, :]
                    imgs = imgs.permute(0, 2, 3, 1)[:]
                    print(
                        "warning: more than 3 channels! only channels 0,1,2 are preserved!"
                    )
                else:
                    raise Exception("unsupported type!  " + str(imgs.size()))
                flag = False
            else:  # single image
                imgs = [imgs]

        if flag:

            def process_img(img):
                if img.dim() == 3:
                    c = img.shape[0]
                    if c == 1:  # grayscale
                        img = img.squeeze()
                    elif c == 3:  # RGB
                        img = img.permute(1, 2, 0)
                    else:
                        raise Exception("unsupported type!  " + str(img.size()))
                img = img.numpy().squeeze()
                return img

            imgs = [process_img(img) for img in imgs]

        if len(imgs) == 1:
            fig, axs = plt.subplots(title=title)
            axs = [axs]
            labels = to_list(labels, k=1)
        else:
            fig, axs = plt.subplots(*grid)
            axs = axs.flatten()
        fig.suptitle(title)  # dunno how to hide figure numbers
        if labels is not None:
            for ax, img, label in zip(axs, imgs, labels):
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                ax.set_title(
                    label if isinstance(label, str) else self.decode_label(label)
                )
        else:
            for ax, img in zip(axs, imgs):
                ax.imshow(img, cmap="gray")
                ax.axis("off")

        plt.tight_layout()

    # expects a list for batch
    def visualize_class(self, class_idx, samples=1):
        train = iter(self.loaders()[0])
        batch_no = 0
        imgs = []
        labels = []
        orig_samples = samples
        while samples > 0:
            batch = next(train)
            inds = torch.nonzero(batch[1] == class_idx)
            num = min(samples, len(inds))
            samples -= num
            for i in range(num):
                imgs.append(batch[0][inds[i]].squeeze(0))
                labels.append(f"(batch {batch_no}, {inds[i].item()})")

            batch_no += 1
        return self.visualize(
            imgs,
            labels,
            grid=factorize(orig_samples),
            title=self.decode_label(class_idx),
        )

