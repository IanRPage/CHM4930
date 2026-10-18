"""Unit tests for the bidirectional LSTM SMILES encoder."""

import pytest
import torch
from torch import nn

from encoders.bilstm_smiles_encoder import (
    PAD_TOKEN,
    UNK_TOKEN,
    biLSTMEncoder,
    build_vocabulary,
    tokenize_smiles,
)


@pytest.fixture
def vocab():
    # hared vocabulary for the encoder tests
    return build_vocabulary(["CCO", "CCCl", "CCBr", "CC(=O)O", "c1ccccc1", "[NH4+]"])


# verify output dimensions across different hidden sizes and layer counts
@pytest.mark.parametrize(
    ("hidden_dim", "num_layers"),
    [(32, 1), (64, 2), (128, 3)],
)
def test_output_shape(vocab, hidden_dim, num_layers):
    encoder = biLSTMEncoder(
        vocab=vocab,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    output = encoder(["CCO", "CCCl", "CC(=O)O"])

    assert encoder.out_dim == hidden_dim * 2
    assert output.shape == (3, hidden_dim * 2)


# a single SMILES string should produce a batch of size one
def test_single_smiles_input(vocab):
    encoder = biLSTMEncoder(vocab=vocab)
    output = encoder("CCO")

    assert output.shape == (1, encoder.out_dim)


# check tokenization of common atoms, bonds, and branching symbols
def test_tokenizer_handles_smiles_symbols():
    assert tokenize_smiles("CCCl") == ["C", "C", "Cl"]
    assert tokenize_smiles("CCBr") == ["C", "C", "Br"]
    assert tokenize_smiles("CC(=O)O") == ["C", "C", "(", "=", "O", ")", "O"]


# ensure bracketed atoms and multi-digit ring identifiers stay intact
def test_tokenizer_handles_brackets_and_rings():
    assert tokenize_smiles("[NH4+]") == ["[NH4+]"]
    assert tokenize_smiles("C%12CCCCC%12") == [
        "C",
        "%12",
        "C",
        "C",
        "C",
        "C",
        "C",
        "%12",
    ]


# vocabulary IDs should be reproducible regardless of training data order
def test_vocabulary_is_deterministic():
    smiles = ["CCO", "CCCl", "CCBr"]

    first = build_vocabulary(smiles)
    second = build_vocabulary(list(reversed(smiles)))

    assert first == second
    assert first[PAD_TOKEN] == 0
    assert first[UNK_TOKEN] == 1
    assert len(first) == len(set(first.values()))


# unseen tokens should map to UNK without causing an encoder failure
def test_unknown_tokens_are_supported():
    vocab = build_vocabulary(["CCO"])
    encoder = biLSTMEncoder(vocab=vocab).eval()

    output = encoder(["CCBr", "[Fe+2]"])

    assert output.shape == (2, encoder.out_dim)
    assert torch.isfinite(output).all()
    assert "Br" not in vocab
    assert "[Fe+2]" not in vocab


# the encoder must support SMILES sequences of different lengths
def test_variable_length_sequences(vocab):
    encoder = biLSTMEncoder(vocab=vocab).eval()

    output = encoder(["C", "CCO", "c1ccccc1"])

    assert output.shape == (3, encoder.out_dim)
    assert torch.isfinite(output).all()


# padding should not change a molecule's embedding
def test_padding_does_not_change_embeddings(vocab):
    encoder = biLSTMEncoder(vocab=vocab).eval()

    single = encoder("CCO")
    batched = encoder(["CCO", "c1ccccc1"])

    assert torch.allclose(single[0], batched[0], atol=1e-6)


# evaluation mode should produce identical outputs across repeated calls
def test_eval_is_deterministic(vocab):
    encoder = biLSTMEncoder(vocab=vocab, dropout=0.5).eval()

    first = encoder(["CCO", "CCCl"])
    second = encoder(["CCO", "CCCl"])

    assert torch.equal(first, second)


# dropout should introduce randomness during training
def test_train_mode_dropout_is_stochastic(vocab):
    encoder = biLSTMEncoder(vocab=vocab, dropout=0.5).train()

    smiles = ["CCO", "CCCl", "CCBr", "CC(=O)O"]
    first = encoder(smiles)
    second = encoder(smiles)

    assert not torch.equal(first, second)


# with dropout disabled, repeated training outputs should match
def test_zero_dropout_is_deterministic(vocab):
    encoder = biLSTMEncoder(vocab=vocab, dropout=0.0).train()

    smiles = ["CCO", "CCCl"]
    assert torch.equal(encoder(smiles), encoder(smiles))


# confirm gradients propagate through every trainable parameter
def test_gradients_reach_every_parameter(vocab):
    encoder = biLSTMEncoder(vocab=vocab)
    encoder(["CCO", "CCCl"]).sum().backward()

    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


# molecular embeddings should be float32 and contain finite values
def test_output_is_float32(vocab):
    encoder = biLSTMEncoder(vocab=vocab)
    output = encoder(["CCO"])

    assert output.dtype == torch.float32


def test_output_is_finite(vocab):
    encoder = biLSTMEncoder(vocab=vocab).eval()
    output = encoder(["CCO", "CCCl", "[NH4+]"])

    assert torch.isfinite(output).all()


# verify the model uses a bidirectional LSTM and reserves padding index zero
def test_uses_bidirectional_lstm(vocab):
    encoder = biLSTMEncoder(vocab=vocab, num_layers=2)

    assert isinstance(encoder.lstm, nn.LSTM)
    assert encoder.lstm.bidirectional
    assert encoder.lstm.num_layers == 2
    assert encoder.embedding.padding_idx == 0


# reject empty, whitespace-containing, or non-string SMILES inputs
@pytest.mark.parametrize("invalid", ["", " ", "C C", 123, None])
def test_rejects_invalid_smiles(invalid):
    with pytest.raises((TypeError, ValueError)):
        tokenize_smiles(invalid)


# reject batches containing no molecules
def test_rejects_empty_batch(vocab):
    encoder = biLSTMEncoder(vocab=vocab)

    with pytest.raises(ValueError, match="empty"):
        encoder([])


# reject invalid dimensions and dropout probabilities
@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": 0},
        {"hidden_dim": 0},
        {"num_layers": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_rejects_invalid_hyperparameters(vocab, kwargs):
    with pytest.raises(ValueError):
        biLSTMEncoder(vocab=vocab, **kwargs)


# reject vocabularies with incorrect reserved token IDs
def test_rejects_invalid_vocabulary():
    invalid_vocab = {PAD_TOKEN: 1, UNK_TOKEN: 0, "C": 2}

    with pytest.raises(ValueError):
        biLSTMEncoder(vocab=invalid_vocab)


# a vocabulary cannot be constructed without training molecules
def test_rejects_empty_training_vocabulary():
    with pytest.raises(ValueError, match="empty"):
        build_vocabulary([])
