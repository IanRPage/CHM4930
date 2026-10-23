"""
End-to-end multimodal fusion model (`MuFu`) combining encoders, projection, fusion, and
heads.
"""

from torch import nn

from mufu.contract import Encoder, MuFuInputs, MuFuOutputs
from mufu.model.fusion import LateFusion
from mufu.model.heads import (
    BioactivityClassificationHead,
    BioactivityRegressionHead,
    ToxicityHead,
)
from mufu.model.modality_dropout import sample_modality_mask
from mufu.model.projection import ModalityProjection


class MuFu(nn.Module):
    """
    Combine three encoder modules, projection, fusion, and prediction heads.

    Construct the encoders separately to choose their weights and settings. Each
    encoder must be an nn.Module with an out_dim attribute. Inputs need only
    smiles, fingerprint, and graph; labels and losses stay outside the model.
    """

    def __init__(
        self,
        smiles_encoder: Encoder,
        fingerprint_encoder: Encoder,
        graph_encoder: Encoder,
    ) -> None:
        super().__init__()
        self.smiles_encoder = smiles_encoder
        self.fingerprint_encoder = fingerprint_encoder
        self.graph_encoder = graph_encoder

        in_dims = {
            "smiles": smiles_encoder.out_dim,
            "fingerprint": fingerprint_encoder.out_dim,
            "graph": graph_encoder.out_dim,
        }
        self.projection = ModalityProjection(in_dims)
        self.fusion = LateFusion()
        self.regression_head = BioactivityRegressionHead()
        self.classification_head = BioactivityClassificationHead()
        self.toxicity_head = ToxicityHead()

    def forward(self, inputs: MuFuInputs) -> MuFuOutputs:
        embeddings = {
            "smiles": self.smiles_encoder(inputs["smiles"]),
            "fingerprint": self.fingerprint_encoder(inputs["fingerprint"]),
            "graph": self.graph_encoder(inputs["graph"]),
        }
        projected = self.projection(embeddings)

        # Mask projected modalities only during training.
        mask = None
        if self.training:
            batch_size = len(inputs["smiles"])
            device = projected["smiles"].device
            mask = sample_modality_mask(batch_size, device=device)

        latent = self.fusion(projected, mask)
        return {
            "pic50": self.regression_head(latent),
            "active_logits": self.classification_head(latent),
            "tox_logits": self.toxicity_head(latent),
        }
