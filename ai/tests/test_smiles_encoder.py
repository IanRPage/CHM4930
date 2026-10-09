from types import SimpleNamespace

import pytest
import torch
from torch import nn

import encoders.smiles_encoder as smiles_module
from encoders.smiles_encoder import SmilesEncoder
from unittest.mock import patch

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

@pytest.mark.parametrize(
    "freeze, expected_grad_enabled",
    [(True, False), (False, True)],
)
def test_gradient_mode_during_forward(
    mocked_pretrained, monkeypatch, freeze, expected_grad_enabled
):
    # check whether gradients are enabled inside MolFormer's forward pass
    _, model = mocked_pretrained
    observed_grad_modes = []

    original_forward = model.forward

    def tracked_forward(*args, **kwargs):
        observed_grad_modes.append(torch.is_grad_enabled())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", tracked_forward)

    encoder = SmilesEncoder(freeze=freeze)
    encoder.eval()

    with torch.enable_grad():
        encoder(["CCO"])

    assert observed_grad_modes == [expected_grad_enabled]

def test_frozen_model_stays_in_eval_mode(mocked_pretrained):
    # a frozen MolFormer should remain in eval mode during training
    encoder = SmilesEncoder(freeze=True)

    encoder.train()

    assert encoder.model.training is False

def test_unfrozen_model_enters_train_mode(mocked_pretrained):
    # an unfrozen MolFormer should support normal training behavior
    encoder = SmilesEncoder(freeze=False)

    encoder.eval()
    encoder.train()

    assert encoder.model.training is True
def test_frozen_molformer_stays_in_eval_when_switching_modes(mocked_pretrained):
    encoder = SmilesEncoder(freeze=True)

    assert encoder.model.training is False

    encoder.train()
    assert encoder.model.training is False

    encoder.eval()
    assert encoder.model.training is False

    encoder.train()
    assert encoder.model.training is False

def test_cache_enabled_by_default_for_frozen_encoder(mocked_pretrained):
    encoder = SmilesEncoder()

    assert encoder.freeze is True
    assert encoder.cache_embeddings is True
    assert encoder._embedding_cache == {}

def test_cached_smiles_skips_model_forward(mocked_pretrained):
    encoder = SmilesEncoder()

    with patch.object(encoder.model, "forward", wraps=encoder.model.forward) as mock_forward:
        first = encoder(["CCO", "CCN"])
        second = encoder(["CCO", "CCN"])

        assert mock_forward.call_count == 1

    assert torch.equal(first, second)
    assert len(encoder._embedding_cache) == 2

def test_duplicate_smiles_in_batch(mocked_pretrained):
    encoder = SmilesEncoder()

    with patch.object(encoder.model, "forward", wraps=encoder.model.forward) as mock_forward:
        result = encoder(["CCO", "CCO", "CCN", "CCO"])

        assert mock_forward.call_count == 1
        assert mock_forward.call_args.kwargs["input_ids"].shape[0] == 2

    assert result.shape == (4, encoder.out_dim)
    assert torch.equal(result[0], result[1])
    assert torch.equal(result[0], result[3])
    assert len(encoder._embedding_cache) == 2

def test_only_missing_smiles_are_encoded(mocked_pretrained):
    encoder = SmilesEncoder()

    with patch.object(encoder.model, "forward", wraps=encoder.model.forward) as mock_forward:
        encoder(["CCO", "CCN"])
        encoder(["CCN", "CCC"])

        assert mock_forward.call_count == 2
        assert mock_forward.call_args.kwargs["input_ids"].shape[0] == 1

    assert len(encoder._embedding_cache) == 3

def test_clear_cache_forces_recomputation(mocked_pretrained):
    encoder = SmilesEncoder()

    with patch.object(encoder.model, "forward", wraps=encoder.model.forward) as mock_forward:
        encoder("CCO")
        encoder("CCO")

        assert mock_forward.call_count == 1

        encoder.clear_cache()
        assert len(encoder._embedding_cache) == 0

        encoder("CCO")
        assert mock_forward.call_count == 2

def test_cache_eviction_preserves_batch_output(mocked_pretrained):
    encoder = SmilesEncoder(max_cache_size=2)

    result = encoder(["CCO", "CCN", "CCC"])

    assert result.shape == (3, encoder.out_dim)
    assert len(encoder._embedding_cache) == 2
    assert "CCO" not in encoder._embedding_cache
    assert "CCN" in encoder._embedding_cache
    assert "CCC" in encoder._embedding_cache

def test_cached_embeddings_preserve_input_order(mocked_pretrained):
    encoder = SmilesEncoder()

    encoder(["CCO", "CCN", "CCC"])

    result = encoder(["CCC", "CCO", "CCN", "CCO"])

    assert torch.equal(result[0], encoder._embedding_cache["CCC"])
    assert torch.equal(result[1], encoder._embedding_cache["CCO"])
    assert torch.equal(result[2], encoder._embedding_cache["CCN"])
    assert torch.equal(result[3], encoder._embedding_cache["CCO"])

def test_cache_rejected_for_trainable_encoder(mocked_pretrained):
    with pytest.raises(ValueError, match="freeze=True"):
        SmilesEncoder(freeze=False, cache_embeddings=True)

def test_cache_disabled_for_trainable_encoder(mocked_pretrained):
    encoder = SmilesEncoder(freeze=False)

    assert encoder.cache_embeddings is False
    assert len(encoder._embedding_cache) == 0

@pytest.mark.parametrize("cache_size", [0, -1])
def test_invalid_cache_size(mocked_pretrained, cache_size):
    with pytest.raises(ValueError, match="max_cache_size"):
        SmilesEncoder(max_cache_size=cache_size)
