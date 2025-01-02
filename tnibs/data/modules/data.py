from dataclasses import field
import os
import torch
from tnibs.data.utils import dfs
from tnibs.config import Config
from tnibs.data.datasets import SubDataset
from tnibs.utils._utils import *
import numpy as np
import torch.utils.data as td
from typing import Any, Dict, List, Optional, Tuple, Union


from tnibs.utils.array import row_index, to_tensors
from tnibs.utils.torch_utils import describe_tensor
from tnibs.utils import Base

class DataConfig(Config):
    data: Optional[Any] = None
    dataset: Optional[Any] = None
    batch_size: int = 32
    num_workers: int = field(default_factory=lambda: os.cpu_count() - 2) # maybe 0 is better default but this more convenient
    sampler: Optional[Any] = None
    transform: Any = None
    shuffle: Optional[bool] = None
    prefetch_factor: tuple[int] = None


# used in wandb
class DataloaderConfig(Config):
    batch_size: int = None
    num_workers: int = None
    sampler: Optional[Any] = None
    shuffle: bool = None
    sampler: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    collate_fn: Optional[Any] = None
    pin_memory: bool = None
    drop_last: bool = None
    timeout: int = None
    worker_init_fn: Optional[Any] = None
    multiprocessing_context: Optional[Any] = None
    generator: Optional[Any] = None
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = None
    pin_memory_device: Optional[str] = None

# dict type datasets are not supported
class Data(Base):
    """The base class of td."""

    def __init__(self):
        super().__init__()

        # For folds
        self._data_inds = None
        self._folds = None
        self._folder = None
        self.processor = None

        # Subclasses define self.data = (X, y), or dataset/(train_set, val_set) directly
        # Be sure to include set_folds

    def _to_dataset(self, data):
        """Converts tensor tuples to dataset"""

        tensors = (to_tensors(x) for x in data)
        return td.TensorDataset(*tensors)

    # prefer not to use this as it doesn't apply to test set
    # by default allow defining scikit processors
    def _fit_transforms(self, tensors):
        if self.processor is not None:
            tensors[0] = self.processor.fit_transform(tensors[0])
            return [lambda x: self.processor.transform(x)]
        else:
            return []

    def _transform(self, tensors, transforms):
        for i, tt in enumerate(zip(tensors, transforms)):
            t, tr = tt
            tensors[i] = tr(t)

    def set_data(
        self,
        *args,
        test=False,
    ):
        data = tuple(
            np.array(a) if isinstance(a, List) else a for a in args
        )  # To allow advanced indexing

        if test:
            self.test_data = data
            self.test_dataset = None
        else:
            self.data = data
            self.dataset = None

    @property
    def data_inds(self):
        if not self._data_inds:
            if self._folds:
                self._data_inds = next(self._folds)
        return self._data_inds or None

    # todo: memory management
    def loaders(
        self,
        train_set=None,
        val_set=None,
        split: Tuple[Union[np.ndarray, List[int]], Union[np.ndarray, List[int]]]
        | None
        | False = None,
        dtypes=[],
        loader_args: List[Dict] = None,
        **kwargs,
    ):
        """returns train_loader, val_loader
        **kwargs: passed to train DataLoader
        split: Tuple of indices that determines how to split the dataset. Prevents splitting if set to False. Controlled by set_folds otherwise.
        loader_args: List of kwargs zipped to loaders. If there are more loaders than loader_args, the last dict is repeated. If not supplied, loader_args=[kwargs].
            - By default, shuffle is False for loaders after the first. If shuffle/sampler is set on self, it will be a default value for the first kwargs only. sampler can be set to True on later args to use the same method as for the first arg.
            - shuffle=True is not set for train if sampler is given.
        """

        def _loaders(*sets):
            nonlocal loader_args
            nonlocal split
            nonlocal kwargs

            # either apply same kwargs to all loaders, or supply a list
            loader_args = loader_args or [kwargs]
            loader_args += [loader_args[-1].copy()] * (len(sets) - len(loader_args))

            # Allow setting some loader args on data class itself
            for ix, la in enumerate(loader_args):
                # Note: batch_sampler is identical to sampler+fixed batch_size
                for key in ["batch_size", "num_workers", "sampler", "shuffle"]:
                    if key not in la.keys():
                        if key == "sampler":
                            if ix == 0:
                                if callable(la.get(key, None)):
                                    la[key] = la[key](self, split[ix])
                                elif getattr(self, key, None):
                                    la[key] = self.sampler(self, split[ix])
                        elif key == "shuffle":
                            if (
                                ix == 0
                                and "sampler" not in la.keys()
                                and not isinstance(sets[ix], td.IterableDataset)
                            ):
                                _shuffle = getattr(self, "shuffle", None)
                                la[key] = (
                                    _shuffle if _shuffle is not None else True
                                )  # shuffle=True shuffles the data after every epoch
                        elif (attr := getattr(self, key, None)) is not None:
                            la[key] = attr

            return tuple(
                td.DataLoader(set, **kwargs) for set, kwargs in zip(sets, loader_args)
            )

        # Allow setting train/val_sets directly
        train_set = (
            train_set
            if isinstance(train_set, td.Dataset)
            else getattr(self, "train_set", None)
        )
        val_set = (
            val_set
            if isinstance(val_set, td.Dataset)
            else getattr(self, "val_set", None)
        )

        if split is None:
            if self.data_inds:
                split = self.data_inds
            else:
                split = False

        train_set = train_set or self.dataset
        if isinstance(train_set, td.Dataset):
            if val_set is not None:
                return _loaders(train_set, val_set)
            elif split is False:
                return _loaders(train_set)
            return _loaders(*(SubDataset(train_set, split[i]) for i in split))

        else:
            # Create Dataset from self.data(*X, y) by converting to tensors
            assert self.data is not None
            if val_set is not None or split is False:
                split = [slice(0, None)]
            dtypes += [torch.float32] * (len(self.data) - len(dtypes))

            train_arr = list(row_index(a, split[0]) for a in self.data)
            # if isinstance(self, ClassifierData):
            #     train_arr[-1] = train_arr[-1].to(torch.int64)
            transforms = self._fit_transforms(train_arr)
            train_set = self._to_dataset(train_arr)

            if val_set is not None:
                return _loaders(train_set, val_set)
            elif split is False:
                return _loaders(train_set)
            else:

                def to_val_set(indices):
                    val_array = [row_index(a, indices) for a in self.data]
                    self._transform(val_array, transforms)
                    return self._to_dataset(val_array)

                return _loaders(train_set, *(to_val_set(ixs) for ixs in split[1:]))

    def test_loader(self, test_set=None, **kwargs):
        test_set = test_set or self.test_dataset or self.test_data
        assert test_set is not None

        if not isinstance(test_set, td.Dataset):
            test_set = self._to_dataset(test_set)

        # Allow setting some loader args on data class itself
        # todo: auto detect best batch size
        for key in ["batch_size", "num_workers"]:
            if (attr := getattr(self, key, None)) is not None:
                kwargs[key] = attr

        return td.DataLoader(test_set, **kwargs)

    def data_range(
        self, loader_index, batch_index, range=None
    ):  # assuming tensors, todo: improve
        loader = iter(self.loaders(shuffle=False)[loader_index])
        outs = next(loader)[batch_index]
        while True:
            try:
                next_batch = next(loader)[batch_index]
                outs = torch.cat((outs, next_batch), dim=0)
            except StopIteration:
                break
        if range:
            return outs[range]
        else:
            return outs

    def set_folds(self, split_method):
        """Configure folds

        Args:
            split_method: from sklearn.model_selection
        """

        self._folder = split_method
        try:
            if self.dataset is not None:
                self._folds = self._folder.split(np.arange(len(self.dataset)))
            if self.data is not None:
                self._folds = self._folder.split(*self.data)
        except AttributeError:
            raise AttributeError("dataset/data not found. Have you called set_data?")

    @property
    def raw_len(self):
        if l := getattr(self, "dataset", None):
            return len(l)
        else:
            try:
                return len(self.data[0])
            except:  # noqa: E722
                raise AttributeError

    # todo: shuffle when train_set is given
    def folds(self, continue_existing=True):
        """Provides an iterator that changes self.data_inds, used by self.loaders(), using iterator provided by self._folds, to be used to loop over folds.

        >>> from sklearn.model_selection import KFold
        >>> t = Data()
        >>> t.data = ([1, 2, 3], [1, 2, 3])
        >>> t.set_folds(KFold(n_splits=3))
        >>> for i in t.folds(): print(t.data_inds)
        (array([1, 2]), array([0]))
        (array([0, 2]), array([1]))
        (array([0, 1]), array([2]))
        """

        def initialize():
            assert self._folder is not None and self.data is not None
            self.set_folds(self._folder)

        if not self._folds or not continue_existing:
            initialize()

        fold_num = 0
        while True:
            try:
                self._data_inds = next(self._folds)
                fold_num += 1
                yield fold_num
            except StopIteration:
                initialize()
                # logging.debug("Reinitializing folds")
                return

    ## Data Preview section

    def sample_batch(
        self,
        batch_index=0,
        train=True,
    ):
        loader = self.loaders()[0 if train else 1]

        for _ in range(batch_index + 1):
            b = next(iter(loader))

        return b

    def preview_batch(self, batch, samples=5, header=True):
        if isinstance(batch, Dict):
            batch_vals = batch.values()
            batch_items = batch.items()
        else:
            batch_vals = batch
            batch_items = enumerate(batch)

        if header:
            # Print shapes for all tensors in the batch
            print("Constituent shapes:")
            for i, tensor in batch_items:
                print(f"batch[{i}]: {tensor.shape}, {tensor.dtype}")

        # Print sample values
        samples = min(samples, len(next(iter(batch_vals))))
        if header:
            print(f"\nFirst {samples} samples:")
        for i in range(samples):
            print(f"\nSample {i}: ")
            for j in batch_vals:
                print(f"\n{j[i].squeeze()}")

    def preview(self, samples=5):
        """
        Preview the data by showing dimensions and sample rows from both training and validation sets.

        Args:
            samples (int): Number of samples to display from each dataset
        """

        def loader_len(loader):
            if isinstance(loader.dataset, td.IterableDataset):
                return "IterableDataset"
            else:
                return f"{len(loader)} batches"

        loaders = self.loaders()

        for i, l in enumerate(loaders):
            print(f"\nLoader {i} ({loader_len(loaders[0])}) Preview:")
            print("-" * 50)
            self.preview_batch(next(iter(l)), samples, header=True)

    def preview_df(self, index):
        dfs.preview_df(self.data[index], head=False)

    def describe_index(
        self, index, batch_index=0, feature_dims=slice(1, None), train=True
    ):
        """
        Provide statistics on features.
        """

        print(f"\n{"Training" if train else "Validation"} Batch[{index}] Statistics:")
        print("-" * 50)
        describe_tensor(
            self.sample_batch(
                batch_index,
                train,
            )[index],
            feature_dims,
        )


# util for concat batches
# def loader_columns(loader, columns=slice(-1, None), cpu=True):
#     outs = [batch[columns] for batch in loader]
#     out = torch.cat(outs, dim=0)
#     if cpu:
#         return out.cpu().numpy()
#     return out
