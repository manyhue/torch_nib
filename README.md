A lean framework for developing, training and testing models. Inspired by the d2l framework of DataModule, Trainer, Module objects, this project builds on that paradigm by generalizing and adding features. Develop, train, and retrain your model in just a couple blocks of readable code!

## Features

- A magic Config class which allows flexible and hassle-free of training hyperparameters and configuration. Only type what you need! Especially useful for hyperparameter search.

- Datamodule wrapping Dataset
  -  Various methods for viewing your data. Know your data *intimately*.
  - Creates train and validation Dataloaders with sensible options.
  - Visualizes image data, provides encoding and decoding for ClassifierData.
  - Easily configure data splitting, and iterating over splits using sklearn cross validators.
  - Autodatset creation from Dataframes and Tensors.

- Some convenient features added to torch modules allowing automatic naming, saving, loading, layer statistics. A pred method allows all evaluators of the classification variant to automatically use the prediction rather than forward output.
- A fully featured training loop preprovisioned methods to activate the following features:
  - Automatic model saving and loading. The `load_previous` kwarg allows Training to resume from saved model/optimizer/scheduler parameters.
  - Auto-gpu discovery, as well as moving the data to the right device for various functions
  - Realtime plotting of loss curves as well as other declarable metrics. Supports both batch and epoch units, depending on whetehr the dataloader is iterable or miniepochs are used.
  - Callbacks for training loop customization
  - Easily log training-time metrics or save them to parquet (todo).
  - A convenient method for evaluating loaded models for one epoch.
  - DistributedDataParallel functionality (IP)
- A MetricFrame class which aggregates metrics compatible with torcheval.
  - Integrates with the infer and fit methods of the Trainer class to allow staggered recording and realtime plotting.
  - Smart units ensure minimal math is needed to determine appropriate parameters

- Convenient utils for data processing, statistic plotting, ndarray manipulation and more


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