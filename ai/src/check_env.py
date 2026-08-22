"""
Quick check to see if the conda environment is set up right

Run `python src/check_env.py` after activating `CHM4930` env
"""

import shutil

import matplotlib
import numpy as np
import pandas as pd
import rdkit
import seaborn as sns
import torch
import torch_geometric

print(f"numpy           {np.__version__}")
print(f"pandas          {pd.__version__}")
print(f"matplotlib      {matplotlib.__version__}")
print(f"seaborn         {sns.__version__}")
print(f"rdkit           {rdkit.__version__}")
print(f"torch           {torch.__version__}")
print(f"torch_geometric {torch_geometric.__version__}")
print(f"jupyter         {'found' if shutil.which('jupyter') else 'NOT FOUND'}")
print(f"CUDA available  {torch.cuda.is_available()}")

print("\ndependencies imported successfully")
