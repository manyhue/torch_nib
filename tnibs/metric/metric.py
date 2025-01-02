import logging
import polars as pl
from tnibs.metric.plot import ProgressBoard
from tnibs.utils.array import to_list
from tnibs.utils import Base

from tnibs.utils._utils import *
import torch

from torcheval.metrics.metric import Metric
from typing import (
    List,
    Optional,
)


# don't squeeze batch index. also tried squeeze(-1) but this may be better
def _default_pred(*args):
    return tuple(a.squeeze(tuple(range(1, len(a.shape)))) for a in args)


# For simplicity, only plotted metrics are kept
class MetricsFrame(Base):
    def __init__(
        self,
        columns: List[Metric],
        flush_every=1,
        name=None,
        xlabel=None,  # not related to board
        unit: Optional[int] = None,
        train=False,  # used to prefix when plotting
        out_device=torch.device("cpu"),
        pred_fun=_default_pred,
        logger=False,
        logging_prefix="",
    ):
        """_summary_

        Args:
            columns (List[Metric]): Instance of torcheval.metrics.metric.Metric with a label property set
            flush_every (int, optional): _description_. Defaults to 1. Flush every n update calls. 0 to never flush.
            name (str, optional): Metric frame name, used for plot title. Can be computed from columns by default.
            xlabel (str, optional): Name of column of xlabel, used for plotting. Defaults to "epoch" or "batch".
            unit: If None, trainer will configure this frame so that all provided values are interpreted in epoch units. None behaves so that unit=num_train_batches in a Trainer with mf_epoch_units=True and 1 otherwise.
            train
            out_device
            pred_fun: defaults to lambda *args: (
                a.squeeze(-1) for a in args
            ), as this is the format most torcheval metrics expect
            logger (Callable | None | False, optional): Will log with self.logger when flush if configured. Set this to None to have Trainer configure this.
        """
        self.save_attr(ignore=["name"])
        self._count = 0

        self._name = name or None

        for c in columns:
            if getattr(c, "label") is None:
                logging.warning(
                    f"{c} does not have a label, imputing from c.__class__.__name__"
                )
                c.label = c.__class__.__name__

        self.dict = {col.label: [] for col in self.columns}
        if xlabel is None:
            self.xlabel = "batch"  # Technically we could use "" or None to denote no xlabel after initialization xlabel is not None, but we use "" to denote no xlabel for clarity
        if self.xlabel:
            self.dict[self.xlabel] = []

        self.df = None
        self.board = None

        self._flush_every = flush_every

    def append(self, *columns: Metric):
        assert all((v == [] for v in self.dict.values())) and self.df is None
        self.columns.extend(columns)
        self.dict = {col.label: [] for col in self.columns}
        if self.xlabel is not None:
            self.dict[self.xlabel] = []

    def rename(self, from_label: str, to_label: str):
        assert all((v == [] for v in self.dict.values())) and self.df is None
        for c in self.columns:
            if c.label == from_label:
                c.label = to_label
                self.dict[to_label] = self.dict.pop(from_label)
                break

    def flush(self, idx=None, log_metric=True, keep_idx=False):
        """Computes and resets all columns.
        If self.logger is configured, will also log computed values (requires xlabel to be set).
        Will use self._count for the idx by default, similar to update(). NOTE that you want
        to be consistent between providing idx or not throughout these two functions.
        Explicitly, using ._count allows us to accumulate statistics across multiple training steps.
        Passing in the index explicitly allows continuing the index from continued training runs.
        That is, choosing between one or the other depends on if you want to
        reuse the metric frame (single graph) vs reuse the setup (multiple graphs).

        Args:
            idx (int): idx to associate with row
            keep_idx (bool): Whether to scale idx by configured units.
        """
        should_log = log_metric and callable(self.logger)
        log_dict = {}

        for c in self.columns:
            val = self.to_out(c.compute())
            self.dict[c.label].append(val)
            if should_log:
                log_dict[self.logging_prefix + c.label] = val
            c.reset()
        if self.xlabel:
            if not keep_idx:
                idx = self.get_idx_per_unit(idx)
            self.dict[self.xlabel].append(idx)
            if should_log:
                log_dict[self.xlabel] = idx
                self.logger(log_dict)

    def compute(self):
        """Computes columns

        Args:
            idx (int): idx to associate with row
        """
        return {c.label: c.compute() for c in self.columns}

    def reset(self):
        for c in self.columns:
            c.reset()

    def clear(self):
        for c in self.columns:
            c.reset()
        self.dict = {col.label: [] for col in self.columns}
        if self.xlabel:
            self.dict[self.xlabel] = []

        self.df = None

    def to(self, device):
        for m in self.columns:
            try:
                m.to(device)
            except:  # noqa: E722
                pass

    def to_out(self, val):
        return to_list(val)

    @property
    def name(self):
        return self._name or ", ".join([col.label for col in self.columns]) + (
            " (training)" if self.train else " (validation)"
        )

    # refactor allows easier override
    def get_idx_per_unit(self, idx):
        if (
            self.unit is None or self.unit == 1
        ):  # Default unit acts as 1, is this inefficient to check...
            return idx or self._count
        else:
            return (idx or self._count) / self.unit

    @torch.inference_mode
    def update(self, *args, idx=None):
        args = self.pred_fun(*args)
        for c in self.columns:
            c.update(*args)

        self._count += 1

        if self._flush_every != 0 and self._count % self._flush_every == 0:
            if not self.xlabel:
                self.flush()
            else:
                self.flush(idx)

    def set_unit(
        self, unit, xlabel, scale_flush_interval=True
    ):  # trainer calls this with xlabel=epoch
        if unit:
            if scale_flush_interval:
                self._flush_every = max(1, int(self.flush_every * unit))
            self.unit = unit
        if xlabel:
            if self.xlabel:
                self.dict[xlabel] = self.dict.pop(self.xlabel)
            else:
                assert all(
                    isinstance(value, list) and not value
                    for value in self.dict.values()
                )  # should not set xlabel for initialized dict
                self.dict[xlabel] = []
            self.xlabel = xlabel

    def init_plot(self, title=None, **kwargs):
        """Convenience method to create a board linked to this to draw on"""
        title = self.name if title is None else title
        if self.board is None:
            logging.info(f"Creating plot for {self.name}")
            self.board = ProgressBoard(xlabel=self.xlabel, title=title, **kwargs)
            self.board.add_mf(self)
        return self.board

    def plot(self, df=False, kind="line"):
        """Plots our graph on our board. If this doesn't work when rerun, you may need to run mf.board.init(). """
        assert self.xlabel
        self.init_plot()

        self.board.ax.clear()  # Clear off other graphs such as that may also be associated to our board.
        if df:  # draw dataframe
            self.board._draw(self.df, self.xlabel, update=True, kind=kind)
        else:
            logging.info(f"Displaying dictionary of {self.name}")
            self.board._draw(self.dict, self.xlabel, update=True, kind=kind)

    def record(self, plot=False):
        if getattr(self, "df") is None:
            self.df = pl.DataFrame(self.dict)

        new_df = pl.DataFrame(self.dict)
        try:
            self.df = self.df.extend(new_df)
            self.dict = {k: [] for k, _ in self.dict.items()}
        except pl.ShapeError:
            logging.info(
                "ShapeError encountered. One or more columns may not compute as scalars. Attempting overwrite of self.df."
            )
            self.df = new_df
            if plot:
                self.plot(df=True)
        return self.df


