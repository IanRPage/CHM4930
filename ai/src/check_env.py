"""
Quick check to see if the conda environment is set up right

Run `python src/check_env.py` after activating `CHM4930` env
"""

import pathlib
import shutil
import subprocess
import sys

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

nbstripout_installed = shutil.which("nbstripout") is not None
nbstripout_registered = (
    subprocess.run(
        ["git", "config", "--get", "filter.nbstripout.clean"],
        capture_output=True,
    ).returncode
    == 0
)
if nbstripout_installed and nbstripout_registered:
    nbstripout_status = "found, registered with git"
elif nbstripout_installed:
    nbstripout_status = "found, but NOT registered - run `nbstripout --install`"
else:
    nbstripout_status = "NOT FOUND"
print(f"nbstripout      {nbstripout_status}")

kernels_dir = pathlib.Path(sys.prefix) / "share" / "jupyter" / "kernels"
# jupyter lowercases kernel names on install
kernel_registered = (kernels_dir / "chm4930" / "kernel.json").exists()
kernel_status = (
    "registered"
    if kernel_registered
    else 'NOT REGISTERED - run `python -m ipykernel install --sys-prefix --name=CHM4930 --display-name="Python (CHM4930)"`'
)
print(f"kernel          {kernel_status}")

print(f"CUDA available  {torch.cuda.is_available()}")

print("\ndependencies imported successfully")
