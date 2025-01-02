
from typing import override
from .data import Data


class LoaderData(Data):
    @override
    def loaders(self):
        return self.train_loader, self.val_loader


