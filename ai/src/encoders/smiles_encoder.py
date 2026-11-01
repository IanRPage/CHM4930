"""
Pretrained MolFormer SMILES sequence encoder with optional embedding caching.
"""

import logging
from collections.abc import Iterable

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"
DEFAULT_REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"
DEFAULT_MAX_CACHE_SIZE = 50_000


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
        freeze: bool = True,
        max_length: int = 202,  # matching MolFormer's max_position_embeddings configuration
        cache_embeddings: bool | None = None,
        max_cache_size: int | None = DEFAULT_MAX_CACHE_SIZE,
        revision: str = DEFAULT_REVISION,
    ) -> None:
        super().__init__()

        if cache_embeddings is None:
            cache_embeddings = freeze

        if cache_embeddings and not freeze:
            raise ValueError("Embedding caches requires freeze=True")

        if max_cache_size is not None and max_cache_size < 1:
            raise ValueError("max_cache_size must be at least 1")

        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self._freeze = freeze
        self.cache_embeddings = cache_embeddings
        self.max_cache_size = max_cache_size
        self._embedding_cache: dict[str, torch.Tensor] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            deterministic_eval=True,
        )

        self.out_dim = self.model.config.hidden_size

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.eval()

    @property
    def freeze(self) -> bool:
        return self._freeze

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze:
            self.model.eval()

        return self

    def clear_cache(self):
        # to remove all previously cached molecular embeddings
        self._embedding_cache.clear()

    def _encode(self, smiles: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        n_truncated = int(
            (encoded["attention_mask"].sum(dim=1) >= self.max_length).sum().item()
        )
        if n_truncated:
            log.warning(
                "truncated %d SMILES reaching max_length=%d",
                n_truncated,
                self.max_length,
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.set_grad_enabled(not self.freeze):
            return self.model(**encoded).pooler_output

    def forward(self, smiles: str | Iterable[str]) -> torch.Tensor:
        if isinstance(smiles, str):
            smiles = [smiles]
        elif isinstance(smiles, Iterable):
            smiles = list(smiles)
        else:
            raise TypeError("expected a SMILES string or an iterable of SMILES strings")

        if len(smiles) == 0:
            raise ValueError("SMILES batch is empty")

        if any(not isinstance(s, str) or not s.strip() for s in smiles):
            raise ValueError("all SMILES inputs must be non-empty strings")

        # without caching, encode the batch normally
        if not self.cache_embeddings:
            return self._encode(smiles)

        # collect cached embeddings needed for this batch
        batch_embeddings = {
            s: self._embedding_cache[s]
            for s in dict.fromkeys(smiles)
            if s in self._embedding_cache
        }
        # identify unique SMILES not already cached
        missing_smiles = [s for s in dict.fromkeys(smiles) if s not in batch_embeddings]

        if missing_smiles:
            new_embeddings = self._encode(missing_smiles).detach().cpu()

            # store embeddings & evict oldest entry if cache is full
            for smiles_string, embedding in zip(missing_smiles, new_embeddings):
                stored_embedding = embedding.clone()
                batch_embeddings[smiles_string] = stored_embedding
                if (
                    self.max_cache_size is not None
                    and len(self._embedding_cache) >= self.max_cache_size
                ):
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
                self._embedding_cache[smiles_string] = stored_embedding
        # reassemble batch in the original order
        param = next(self.model.parameters())
        return torch.stack([batch_embeddings[s] for s in smiles]).to(
            device=param.device, dtype=param.dtype
        )
