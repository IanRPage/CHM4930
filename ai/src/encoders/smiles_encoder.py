from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"


class SmilesEncoder(nn.Module):
    """
    Pretrained MolFormer encoder for SMILES strings; takes either a single SMILES string or
    a batch of SMILES strings and returns one pooled molecular representation per molecule. A
    single SMILES string is treated as a batch of size 1.

    By default, the pretrained MolFormer weights are frozen.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        freeze: bool = True,  # change to a method later to allow explicit freeze/unfreeze control? consider it
        max_length: int = 202,  # matching MolFormer's mox_position_embdeddings configuration
        cache_embeddings: bool | None = None,
        max_cache_size: int = 10000,
    ) -> None:
        super().__init__()

        if cache_embeddings is None:
            cache_embeddings = freeze

        if cache_embeddings and not freeze:
            raise ValueError("Embedding caches requires freeze=True")

        if max_cache_size < 1:
            raise ValueError("max_cache_size must be at least 1")

        self.model_name = model_name
        self.max_length = max_length
        self.freeze = freeze
        self.cache_embeddings = cache_embeddings
        self.max_cache_size = max_cache_size
        self._embedding_cache: dict[str, torch.Tensor] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            deterministic_eval=True,
        )

        self.out_dim = (
            self.model.config.hidden_size
        )  # will be default 768; consider 512 later to match fingerprint

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze:
            self.model.eval()

        return self

    def clear_cache(self):
        # to remove all previously cached molecular embeddings
        self._embedding_cache.clear()

    def forward(self, smiles: str | Sequence[str]) -> torch.Tensor:
        if isinstance(smiles, str):
            smiles = [smiles]
        elif not isinstance(smiles, Sequence):
            raise TypeError("expected a SMILES string or a sequence of SMILES strings")

        if len(smiles) == 0:
            raise ValueError("SMILES batch is empty")

        if any(not isinstance(s, str) or not s.strip() for s in smiles):
            raise ValueError("all SMILES inputs must be non-empty strings")

        device = next(self.model.parameters()).device

        # without caching, encode the batch normally
        if not self.cache_embeddings:
            encoded = self.tokenizer(
                list(smiles),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            if self.freeze:
                with torch.no_grad():
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)

            return outputs.pooler_output

        # collect cached embeddings needed for this batch
        batch_embeddings = {
            s: self._embedding_cache[s]
            for s in dict.fromkeys(smiles)
            if s in self._embedding_cache
        }
        # identify unique SMILES not already cached
        missing_smiles = [s for s in dict.fromkeys(smiles) if s not in batch_embeddings]

        if missing_smiles:
            encoded = self.tokenizer(
                missing_smiles,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)

            new_embeddings = outputs.pooler_output.detach().cpu()

            # store embeddings & evict oldest entry if cache is full
            for smiles_string, embedding in zip(missing_smiles, new_embeddings):
                stored_embedding = embedding.clone()
                batch_embeddings[smiles_string] = stored_embedding
                if len(self._embedding_cache) >= self.max_cache_size:
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
                self._embedding_cache[smiles_string] = stored_embedding
        # reassemble batch in the original order
        return torch.stack([batch_embeddings[s].to(device) for s in smiles])
