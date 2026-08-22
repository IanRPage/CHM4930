# CHM4930: AI

This is where all code relating to training our deep learning models will go,
along with any datasets or checkpoints for saved model weights.

## Directory Structure

```
ai/
├── data/            # local data storage (might store this way?)
│   ├── raw/
│   └── processed/
├── notebooks/       # exploration / scratchpads
├── src/             # Main code
│   ├── dataset.py   # custom dataset & dataLoader logic
│   ├── model.py     # PyTorch network architectures
│   ├── train.py     # training loops & validation
│   └── evaluate.py  # metrics, confusion matrices, evaluation
├── checkpoints/     # saved model weights
├── outputs/         # generated artifacts (figures, logs, metrics, etc.)
├── environment.yml  # conda environment config
└── README.md
```

## Configuring Environment

We chose to use Conda as it's the simplest for setting up an environment while
still be cross compatible with different OSes and different hardware
configurations. Make sure you have
[`conda`](https://www.anaconda.com/docs/getting-started/installation) installed
on your system.

To build the environment everything will be run in, make sure you're in the root
of this directory and run

```
conda env create -f environment.yml
```

From there, just activate the environment

```
conda activate CHM4930
```

And that's it, you're good to start doing stuff! When you're done, deactivate
the environment by running

```
conda deactivate
```
