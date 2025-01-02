from typing import Dict, Iterable, TYPE_CHECKING
import IPython.display
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tnibs.utils import Base, is_notebook, vb


if TYPE_CHECKING:
    from tnibs.metric.metric import MetricsFrame


# todo: set title and y_label
class ProgressBoard(Base):
    def __init__(
        self,
        width=800,
        height=600,
        xlim=None,
        ylim=None,
        title=None,
        xlabel="X",
        ylabel="Y",
        xscale="linear",
        yscale="linear",
        labels=None,
        display=True,
        draw_every=5,  # draw every n epochs
        update_df_every=5,  # update df ever n points
        interactive=None,
        save=False,
        train_prefix="train/",
    ):
        self.save_attr()

        ## Graphical params
        sns.set_style("whitegrid")

        self.init_plot()
        self.mfs = []
        self._count = 0

        # See draw points
        ## Initialize data structures
        # assert draw_every >= 1

        # if labels:
        #     self.schema = pl.Schema(
        #         [(xlabel, pl.Float64), (ylabel, pl.Float64), ("Label", pl.Enum(labels))]
        #     )

        # else:
        #     self.schema = pl.Schema(
        #         [(xlabel, pl.Float64), (ylabel, pl.Float64), ("Label", pl.String())]
        #     )

        # self._points_count = 0

        # self.data = pl.DataFrame(
        #     schema=self.schema, data=[]
        # )  # To store processed data (mean of every n)

        # self._clear_buffer()

        ## Further config

        # legend_labels = []
        # for orbit in self.data['Label'].unique():
        #     legend_labels.append(f"{orbit}")

        # handles, _ = self.ax.get_legend_handles_labels()
        # self.ax.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(1.01, 0.29), title="Orbit")

    def init_plot(self):
        if (fig := getattr(self, "fig", None)) is not None:
            plt.close(fig)
        self.fig, self.ax = plt.subplots(
            figsize=(self.width / 100, self.height / 100)
        )  # Adjust size for Matplotlib

        if not isinstance(self.interactive, bool):
            self.interactive = is_notebook()
        if self.interactive:
            self.dh = IPython.display.display(self.fig, display_id=True)
            plt.close()  # ipython ipdate handles the plot

        # Set log scaling based on the provided xscale and yscale
        if self.xscale == "log":
            self.ax.set_xscale("log")
        if self.yscale == "log":
            self.ax.set_yscale("log")
        if self.title:
            self.ax.set_title(self.title)

    def _redef_ax(self):
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.set_yscale(self.yscale)

    def _draw(self, data, xlabel, update=False, kind="line"):
        # there should be definite way to plot dataframes
        if isinstance(data, pl.DataFrame):
            for col in data.columns:
                if col != xlabel:
                    if kind == "scatter":
                        sns.scatterplot(
                            x=xlabel,
                            y=data[col],
                            label=col,
                            data=data,
                            ax=self.ax,
                        )
                    else:
                        sns.lineplot(
                            x=xlabel,
                            y=data[col],
                            label=col,
                            data=data,
                            ax=self.ax,
                        )
        elif isinstance(data, Dict):
            for i, col in enumerate(data.keys()):
                if col != xlabel and data[col]:
                    if kind == "scatter":
                        sns.scatterplot(
                            x=xlabel,
                            y=col,
                            label=col,
                            data=data,
                            ax=self.ax,
                        )
                    else:
                        sns.lineplot(
                            x=xlabel,
                            y=col,
                            label=col,
                            data=data,
                            # not sure how to handle overlapping lines
                            linewidth=1.5 - 0.2 * i, 
                            ax=self.ax,
                        )
        else:
            raise TypeError

        if update:
            self._redef_ax()
            self.iupdate()

    def draw_mfs(self, force=False):
        self._count += 1

        if not force and (not self.display or self._count % self.draw_every) != 0:
            return
        self.ax.clear()

        for mf in self.mfs:
            if mf.train:
                self._draw(
                    {
                        self.train_prefix + key if key != mf.xlabel else key: value
                        for key, value in mf.dict.items()
                    },
                    mf.xlabel,
                )
            else:
                self._draw(mf.dict, mf.xlabel)
        self._redef_ax()
        self.iupdate()

    def add_mf(self, *mfs: Iterable["MetricsFrame"]):
        for mf in mfs:
            self.mfs.append(mf)
            if mf.board is None:
                mf.board = self

    def close(self):
        plt.close()

    def flush(self):
        for key in self.raw_points.keys():
            self.draw([], [], key, force=True)
        self.draw_mfs(force=True)
        if self.save:
            plt.savefig("updated_plot.png")

    def iupdate(self):
        if self.interactive:
            self.dh.update(self.fig)

    # todo
    # def _clear_buffer(self):
    #     self.buffer = {k: [] for k in (self.xlim, self.ylim, "Labels")}

    # # todo: improved aggregation
    # def draw_points(self, x, y, label, every_n=5, force=False, clear=False):
    #     """Update plot with new points (arrays) and redraw."""

    #     self.buffer[self.xlim].append(x)
    #     self.buffer[self.ylim].append(x)
    #     self.buffer["Labels"].append(label)

    #     if len(self.buffer["Labels"]) >= self.update_df_every or force:
    #         new_df = pl.DataFrame(self.buffer)
    #         self.data = self.data.extend(new_df)
    #         self._clear_buffer()

    #     if not self.display:
    #         return

    #     # # X-axis values (common for all lines)
    #     # x_values = [0, 1, 2, 3, 4]

    #     # # Plot using the dictionary directly
    #     # sns.lineplot(data=data, palette='tab10')

    #     # # Setting x-values explicitly
    #     # plt.xticks(ticks=range(len(x_values)), labels=x_values)
    #     # Redraw the plot

    #     if True:
    #         if clear:
    #             self.ax.clear()
    #             sns.scatterplot(x=self.xlabel, y=self.ylabel, hue="Label", data=new_df)
    #         else:
    #             sns.scatterplot(
    #                 x=self.xlabel, y=self.ylabel, hue="Label", data=self.data
    #             )

    #     else:
    #         for label in self.labels:
    #             label_data = self.data.filter(pl.col("Label") == label)
    #             sns.lineplot(
    #                 x="X",
    #                 y="Y",
    #                 data=label_data,
    #                 ax=self.ax,
    #                 label=label,
    #                 linestyle=self.line_styles[label],
    #                 color=self.line_colors[label],
    #             )

    #     self.iupdate()



# plot 2d array
def plot_2dheatmap(arr, ticklabels="auto", **kwargs):
    # example showing plt.figure is the same as plt.subplots, with ax = plt.gca()
    plt.figure()
    sns.heatmap(
        [[int(el) for el in row] for row in arr],
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        ax=plt.gca(),
    )
    kwargs.setdefault("xlabel", "Predicted")
    kwargs.setdefault("ylabel", "Ground truth")
    plt.gca().set(**kwargs)
    plt.show()


def plot_hist(list, kde=True, **kwargs):
    plt.figure()
    binwidth = 1 if isinstance(list[0], int) else None
    sns.histplot(list, binwidth=binwidth, kde=kde)
    kwargs.setdefault("xlabel", "Value")
    kwargs.setdefault("ylabel", "Count")
    plt.gca().set(**kwargs)


def plot_bar(
    dict, stack_label="variant", xlabel="class", ylabel="count", x_rot=0, **kwargs
):
    fig, ax = plt.subplots()

    if isinstance(next(iter(dict.keys())), tuple):
        df = pd.DataFrame(list(dict.items()), columns=["Tuple", "Count"])
        df[["a", stack_label]] = pd.DataFrame(df["Tuple"].tolist(), index=df.index)

        # Pivot the DataFrame to get counts by 'a' and 'b'
        pivot_df = df.pivot_table(
            index="a", columns=stack_label, values="Count", aggfunc="sum", fill_value=0
        )
        # Plot a stacked bar chart
        pivot_df.plot(kind="bar", stacked=True, rot=x_rot, ax=ax)
        # ax.tick_params(axis="x", rotation=0) doesn't work
    else:
        sns.barplot(dict)

    kwargs.setdefault("xlabel", "Class")
    kwargs.setdefault("ylabel", "Count")
    ax.set(**kwargs)
    plt.show()

def plot_points(
    *point_lists,
    kind="scatter",  # or line
    default_time=None,
    set_labels=None,
    **kwargs,
):
    fig, ax = plt.subplots()
    kwargs.setdefault("xlabel", "Value")
    kwargs.setdefault("ylabel", "Index")
    legend = "Legend"

    all_data = []

    set_labels = set_labels or [f"Set{i}" for i in range(len(point_lists))]

    for point_list, label in zip(point_lists, set_labels):
        if len(point_list) == 0:
            if vb(7):
                print("label is empty")
            continue
        if isinstance(point_list[0], tuple):
            data_points, time_values = zip(*point_list)
            df = pd.DataFrame(
                {
                    kwargs["xlabel"]: time_values,
                    kwargs["ylabel"]: data_points,
                    legend: [label] * len(point_list),
                }
            )
        else:
            time = (
                default_time[: len(point_list)]
                if default_time is not None
                else np.arange(1, len(point_list) + 1)
            )
            df = pd.DataFrame(
                {
                    kwargs["xlabel"]: time,
                    kwargs["ylabel"]: point_list,
                    legend: [label] * len(point_list),
                }
            )

        all_data.append(df)

    combined_df = pd.concat(all_data)

    # Plot using Seaborn
    if kind == "scatter":
        sns.scatterplot(
            data=combined_df,
            x=kwargs["xlabel"],
            y=kwargs["ylabel"],
            hue=legend,
            style=legend,
            ax=ax,
        )  # s=100 for size
    else:
        sns.lineplot(
            data=combined_df,
            x=kwargs["xlabel"],
            y=kwargs["ylabel"],
            hue=legend,
            style=legend,
            ax=ax,
        )
    # alternatively relplot creates its own figure

    ax.set(**kwargs)

    # Show the plot
    plt.show()