from types import SimpleNamespace

import pytest
import encoders.smiles_encoder as smiles_module
import torch
from encoders.smiles_encoder import SmilesEncoder
from torch import nn

HIDDEN_SIZE = 32

"""
This test file tests the wrapper logic of the smiles encoder; the actual pretrained MolFormer is tested separately in a notebook.
"""


class DummyTokenizer:
    # lightweight stand-in for the MolFormer tokenizer for unit tests
    def __init__(self):
        self.last_smiles = None
        self.last_kwargs = None

    def __call__(self, smiles, **kwargs):
        self.last_smiles = smiles
        self.last_kwargs = kwargs

        batch_size = len(smiles)
        seq_len = 4

        return {
            "input_ids": torch.ones(
                batch_size,
                seq_len,
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                batch_size,
                seq_len,
                dtype=torch.long,
            ),
        }


class DummyMolFormer(nn.Module):
    # small trainable model w/ the input the smiles encoder expects
    def __init__(self):
        super().__init__()

        self.config = SimpleNamespace(hidden_size=HIDDEN_SIZE)
        self.embedding = nn.Embedding(8, HIDDEN_SIZE)
        self.projection = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        pooled = self.projection(x.mean(dim=1))

        return SimpleNamespace(pooler_output=pooled)


@pytest.fixture
def mocked_pretrained(monkeypatch):
    # replace Hugging Face loading with lightweight local test objects
    tokenizer = DummyTokenizer()
    model = DummyMolFormer()

    monkeypatch.setattr(
        smiles_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    monkeypatch.setattr(
        smiles_module.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )

    return tokenizer, model


# tests that a batch produces one embedding per molecule with the advertised size
def test_output_shape_for_batch(mocked_pretrained):
    encoder = SmilesEncoder()

    output = encoder(
        [
            "CCO",
            "CC(=O)O",
            "c1ccccc1",
        ]
    )

    assert encoder.out_dim == HIDDEN_SIZE
    assert output.shape == (3, HIDDEN_SIZE)


# tests that a single SMILES string is supported and treated as a batch of size 1
def test_single_string_returns_batch_of_one(mocked_pretrained):
    encoder = SmilesEncoder()

    output = encoder("CCO")

    assert output.shape == (1, HIDDEN_SIZE)


# tests that "CCO" and ["CCO"] follow equivalent encoding behavior
def test_single_string_matches_single_item_batch(mocked_pretrained):
    encoder = SmilesEncoder().eval()

    with torch.no_grad():
        single_output = encoder("CCO")
        batch_output = encoder(["CCO"])

    assert torch.equal(single_output, batch_output)


# tests that encoder outputs contain valid finite floating-point values
def test_output_is_float32_and_finite(mocked_pretrained):
    encoder = SmilesEncoder()

    output = encoder(["CCO", "CCN"])

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


# tests that evaluation mode gives repeatable embeddings for identical input
def test_eval_is_deterministic(mocked_pretrained):
    encoder = SmilesEncoder().eval()

    with torch.no_grad():
        output1 = encoder(["CCO", "CCN"])
        output2 = encoder(["CCO", "CCN"])

    assert torch.equal(output1, output2)


# tests that an empty batch is rejected instead of silently producing bad output
def test_empty_batch_rejected(mocked_pretrained):
    encoder = SmilesEncoder()

    with pytest.raises(ValueError, match="empty"):
        encoder([])


# tests that empty, whitespace-only, and non-string batch entries are rejected
@pytest.mark.parametrize(
    "smiles",
    [
        [""],
        ["   "],
        ["CCO", ""],
        ["CCO", None],
        ["CCO", 123],
    ],
)
def test_invalid_batch_entries_rejected(mocked_pretrained, smiles):
    encoder = SmilesEncoder()

    with pytest.raises(ValueError, match="non-empty strings"):
        encoder(smiles)


# tests that unsupported input types fail with a clear error
@pytest.mark.parametrize(
    "invalid_input",
    [
        123,
        3.14,
        None,
    ],
)
def test_non_string_non_sequence_input_rejected(
    mocked_pretrained,
    invalid_input,
):
    encoder = SmilesEncoder()

    with pytest.raises(TypeError):
        encoder(invalid_input)


# tests that pretrained MolFormer parameters are frozen by default
def test_frozen_by_default(mocked_pretrained):
    encoder = SmilesEncoder()

    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())


# tests that freeze=False leaves MolFormer parameters trainable for fine-tuning
def test_freeze_false_keeps_model_trainable(mocked_pretrained):
    encoder = SmilesEncoder(freeze=False)

    assert all(parameter.requires_grad for parameter in encoder.model.parameters())


# tests that gradients can propagate through MolFormer when it is unfrozen
def test_gradients_reach_model_when_unfrozen(mocked_pretrained):
    encoder = SmilesEncoder(freeze=False)

    output = encoder(["CCO", "CCN"])
    output.sum().backward()

    for name, parameter in encoder.model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


# tests that out_dim accurately reports the underlying model's embedding width
def test_out_dim_matches_model_hidden_size(mocked_pretrained):
    encoder = SmilesEncoder()

    assert encoder.out_dim == encoder.model.config.hidden_size


# tests that tokenization uses the padding, truncation, and max-length settings
# required to safely process variable-length SMILES batches
def test_tokenizer_configuration(mocked_pretrained):
    tokenizer, _ = mocked_pretrained

    encoder = SmilesEncoder(max_length=128)
    encoder(["CCO", "c1ccccc1"])

    assert tokenizer.last_kwargs["padding"] is True
    assert tokenizer.last_kwargs["truncation"] is True
    assert tokenizer.last_kwargs["max_length"] == 128
    assert tokenizer.last_kwargs["return_tensors"] == "pt"


# tests that single-string input is normalized before reaching the tokenizer
def test_single_string_is_converted_to_batch(mocked_pretrained):
    tokenizer, _ = mocked_pretrained

    encoder = SmilesEncoder()
    encoder("CCO")

    assert tokenizer.last_smiles == ["CCO"]
