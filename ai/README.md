# CHM4930: AI

This is where all code relating to training our deep learning models will go,
along with any datasets or checkpoints for saved model weights.

## Directory Structure

```
ai/
├── data/            # local data storage
├── notebooks/       # exploration / scratchpads
├── src/             # main code
│   └── check_env.py # sanity check that dependencies are installed correctly
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

To make sure everything installed correctly, run the enviroment check

```
python src/check_env.py
```

It imports each dependency and prints its version, so if something's missing or
broken, you'll know
