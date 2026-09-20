import collections
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from deepchem.molnet.load_function.clintox_datasets import load_clintox
from deepchem.molnet.load_function.tox21_datasets import load_tox21
from deepchem.data import NumpyDataset
from deepchem.data.datasets import DiskDataset
from deepchem.feat import RawFeaturizer
from deepchem.utils.data_utils import save_to_disk, load_from_disk
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem, RDLogger

'''
load_clintox/load_tox21 output: Tuple[List[str], Tuple[Dataset, ...], List[dc.trans Transformer]
   - Dataset type is deepchem.data.datasets.DiskDataset

load_clintox/load_tox21 parameters to note:

splitter: Splitter or str
        the splitter to use for splitting the data into training, validation, and
        test sets.  Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut.  If this is None, all the data
        will be included in a single dataset.

transformers: list of TransformerGenerators or strings
        the Transformers to apply to the data.  Each one is specified by a
        TransformerGenerator or, as a shortcut, one of the names from
        dc.molnet.transformers.

'''

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()

def standardize_smiles(smiles: str) -> str | None:
   mol = Chem.MolFromSmiles(smiles)

   if mol is None:
      return None

   mol = _LARGEST_FRAGMENT.choose(mol)

   return Chem.MolToSmiles(mol)


def clean_dataset(toxicity_data, duplicate_mode):
   valid_indices = []
   standardized_smiles = []

   # maps standardized SMILES -> list of original dataset indices
   smiles_to_indices = {}

   for i, smiles in enumerate(toxicity_data.ids):
      standardized = standardize_smiles(smiles)

      if standardized is None:
         continue

      if standardized not in smiles_to_indices:
         smiles_to_indices[standardized] = []

      smiles_to_indices[standardized].append(i)

   for standardized, indices in smiles_to_indices.items():
      # keeping first duplicate instance for tox21
      if len(indices) == 1 or duplicate_mode == "keep_first":
         valid_indices.append(indices[0])
         standardized_smiles.append(standardized)

      # remove all duplicated for clintox
      elif duplicate_mode == "remove_all":
         continue
        
   toxicity_data_filtered = toxicity_data.select(valid_indices)

   toxicity_data_filtered = NumpyDataset(
      X=toxicity_data_filtered.X,
      y=toxicity_data_filtered.y,
      w=toxicity_data_filtered.w,
      ids=standardized_smiles
   )

   return toxicity_data_filtered

def cache_dataset(dataset, cache_path):
   cache_path.parent.mkdir(parents=True, exist_ok=True)

   df = pd.DataFrame({
      "smiles": dataset.ids,
   })

   for i in range(dataset.x.shape[1]):
      df[f"x_{i}"] = dataset.x[:, i]
      df[f"y_{i}"] = dataset.y[:, i]
      df[f"w_{i}"] = dataset.w[:, i]

   df.to_csv(cache_path, index=False)
   log.info("wrote %d molecules to %s", len(dataset), cache_path)

def load_toxicity_data(dataset_name: str) -> DiskDataset:
   cache_path = Path(DATA_DIR / f"MolNet-{dataset_name}")
   if not cache_path.exists():
      # pull data without splitting, set molecular representation to SMILES
      if dataset_name == "clintox":
         dataset = load_clintox(splitter=None, featurizer=RawFeaturizer(smiles=True))
         dataset_clean = clean_dataset(dataset[1][0], "remove_all")
         print(f"\n{len(dataset_clean)} molecules from clintox")
      elif dataset_name == "tox21":
         dataset = load_tox21(splitter=None, featurizer=RawFeaturizer(smiles=True))
         dataset_clean = clean_dataset(dataset[1][0], "keep_first")
         print(f"\n{len(dataset_clean)} molecules from tox21")

      tasks = dataset[0]
      transformers = dataset[2]
      
      # save cleaned dataset
      dataset = DiskDataset.from_numpy(
         X=dataset_clean.X,
         y=dataset_clean.y,
         w=dataset_clean.w,
         ids=dataset_clean.ids,
         tasks=tasks,
         data_dir=str(cache_path),
      )

      # save task names
      with open(cache_path / "tasks.json", "w") as f:
         json.dump(tasks, f)

      # save tranformers
      save_to_disk(transformers,str(cache_path / "transformers.joblib"))
      
      return (tasks,(dataset,),transformers)
   else:
      log.info("using cached %s", cache_path)
      with open(cache_path / "tasks.json") as f:
         tasks = json.load(f)

      dataset = DiskDataset(str(cache_path))

      transformers = load_from_disk(str(cache_path / "transformers.joblib"))

      return (tasks,(dataset,),transformers)


def main() -> None:
   RDLogger.DisableLog("rdApp.info")
   clintox_clean = load_toxicity_data("clintox")
   print(clintox_clean)
   tox21_clean = load_toxicity_data("tox21")
   print(tox21_clean)
   
   
if __name__ == "__main__":
   main()