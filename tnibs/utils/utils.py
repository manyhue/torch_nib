import inspect
from contextlib import contextmanager
import os
import torch


from typing import Dict, List, TYPE_CHECKING

from tnibs.config import Config


# main utils, useful for user and implementation


def is_notebook():
    # credit -> https://stackoverflow.com/a/39662359
    try:
        shell = get_ipython().__class__.__name__ # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

class Base:
    def save_attr(self, ignore=[], clobber=True, expand_kwargs=True, save_c=False):
        """Save function arguments into class attributes. Prefer save_config"""

        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        config = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }

        if expand_kwargs:
            kwargs = config.pop("kwargs", None)
            if isinstance(kwargs, Dict):
                for k, v in kwargs.items():
                    if k not in set(ignore + ["self"]) and not k.startswith("_"):
                        config[k] = v
        
        if save_c:
            assert "c" not in config.keys()
            self.c=config

        for k, v in config.items():
            if clobber or getattr(self, k, None) is None:
                setattr(self, k, v)

    def save_config(self, c: Config, ignore=[], clobber=True):
        """save_attr but with a config object"""
        for k, v in c.__dict__.items():
            if k not in ignore:
                if clobber or getattr(self, k, None) is None:
                    setattr(self, k, v)
        self.c = c

    # We get gpus with get_gpus, this allows getting the main gpu
    @property
    def device(self):
        try:
            return self.gpus[0]
        except:  # noqa: E722
            return "cpu"

def get_gpus(gpus: int | List[int] = -1, vendor="cuda"):
    """Given num_gpus or array of ids, returns a list of torch devices

    Args:
        gpus (int | List[int], optional): [] for cpu. Defaults to -1 for all gpus.
        vendor (str, optional): vendor_string. Defaults to "cuda".

    Returns:
        _type_: _description_
    """
    if isinstance(gpus, list):
        assert [int(i) for i in gpus]
    elif gpus == -1:
        gpus = range(torch.cuda.device_count())
    else:
        assert gpus <= torch.cuda.device_count()
        gpus = range(gpus)
    return [torch.device(f"{vendor}:{i}") for i in gpus]



@contextmanager
def change_dir(target_dir):
    original_dir = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)


def vb(n):
    """Shorthand to allow: if vb(n): print()"""
    try:
        return verbosity >= n  # type: ignore
    except NameError:
        return False

def dbg(*args):
    frame = inspect.currentframe().f_back
    print(f"funcname = {frame.f_code.co_name} -", *args)






