from collections.abc import Sequence
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"

class SmilesEncoder(nn.Module):
    """
    Pretrained MolFormer encoder for SMILES strings; takes a batch of SMILES strings and 
    returns one pooled molecular representation per molecule. 

    By default, the pretrained MolFormer weights are frozen.
    """

    def __init__(
            self, 
            model_name: str = DEFAULT_MODEL_NAME, 
            freeze: bool = True, #change to a method later to allow explicit freeze/unfreeze control? consider it
            max_length: int = 202, 
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code = True,
        )

        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code = True, 
            deterministic_eval = True,
        )

        self.out_dim = self.model.config.hidden_size # will be default 768; consider 512 later to match fingerprint

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(self, smiles: Sequence[str]) -> torch.Tensor:
        if isinstance(smiles, str):
            raise TypeError("expected a batch of SMILES strings, not a single string")
       
        if not smiles:
            raise ValueError("SMILES batch is empty")
        
        if any(not isinstance(s, str) or not s.strip() for s in smiles):
            raise ValueError("all SMILES inputs must be non-empty strings")

        device = next(self.model.parameters()).device

        encoded = self.tokenizer(
            list(smiles), 
            padding = True, 
            truncation = True, 
            max_length = self.max_length, 
            return_tensors = "pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = self.model(**encoded)

        return outputs.pooler_output