A lean framework for developing, training, and testing models. Inspired by the D2L framework of DataModule, Trainer, and Module objects, this project builds on that paradigm by generalizing and adding features. Develop, train, and retrain your model in just a couple of blocks of code.

## Features

- A magic `Config` class allows a simple and flexible way of working with training hyperparameters and configuration. Only type what you need! Useful for hyperparameter searches.

- `DataModule` wrapping `Dataset`:
  - Various methods for viewing your data. Know your data *intimately*.
  - Creates train and validation `DataLoaders` with sensible options.
  - Visualizes image data, provides encoding and decoding for `ClassifierData`.
  - Easily configure data splitting and iterate over splits using sklearn cross-validators.
  - Auto-dataset creation from DataFrames and tensors.

- Some convenient features added to Torch modules allow automatic naming, saving, loading, and layer statistics. A `pred` method ensures all evaluators of the classification variant automatically use the prediction rather than forward output.

- A training loop:
  - Automatic model saving and loading. The `load_previous` kwarg allows training to resume from saved model/optimizer/scheduler parameters.
  - Auto-GPU discovery, as well as moving the data to the correct device for various functions.
  - Real-time plotting of loss curves and other declarable metrics. Supports both batch and epoch units, depending on whether the DataLoader is iterable or mini-epochs are used.
  - Callbacks for training loop customization.
  - Easily log training-time metrics or save them to Parquet (IP).
  - A convenient method for evaluating loaded models for one epoch.
  - `DistributedDataParallel` functionality (IP).

- A `MetricFrame` class aggregates metrics compatible with `torcheval`:
  - Integrates with the `infer` and `fit` methods of the `Trainer` class to allow staggered recording and real-time plotting.
  - Interprets the configured "unit" as epoch or batch units depending on training or inference.

- Convenient utilities for data processing, statistic plotting, ndarray manipulation, and more.

- Check out [chess_seq_probe](https://github.com/manyhue/seq_chess_probe) and the examples folder (IP).

## Installation

You can install `torch-nibs` via pip:

```bash
pip install torch-nibs
```


Note that this project was created with pixi. If you are installing with pip, you will also need the following dependencies:

```
"torch>=1.10.0",
"torchvision>=0.11.0",
"torchaudio>=0.10.0",
"polars",
"wandb",
"jupyter",
"pip",
"ipympl",
"plotly",
"tqdm",
"seaborn>=0.13.2,<0.14",
"scikit-learn>=1.5.2,<2",
"openpyxl>=3.1.5,<4",
"fastexcel>=0.12.0,<0.13",
"pandas>=2.2.3,<3",
"datasets>=3.2.0,<4; extra == 'huggingface'",
"jupyter_console>=6.6.3,<7",

```