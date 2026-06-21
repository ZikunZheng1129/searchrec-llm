import torch

from src.sequence_models.gru4rec import GRU4Rec
from src.sequence_models.sasrec import SASRec


def test_gru4rec_forward_and_scores_shape() -> None:
    model = GRU4Rec(num_items=5, embedding_dim=8, hidden_dim=8)
    input_ids = torch.tensor([[0, 1, 2], [0, 3, 4]])

    user_repr = model(input_ids)
    scores = model.score_items(user_repr)

    assert user_repr.shape == (2, 8)
    assert scores.shape == (2, 6)


def test_gru4rec_recommend_masks_padding_and_seen() -> None:
    model = GRU4Rec(num_items=5, embedding_dim=8, hidden_dim=8)
    input_ids = torch.tensor([[0, 1, 2]])
    exclude_ids = torch.tensor([[1, 2]])

    _, item_ids = model.recommend(input_ids, top_k=3, exclude_ids=exclude_ids)

    assert 0 not in item_ids.tolist()[0]
    assert 1 not in item_ids.tolist()[0]
    assert 2 not in item_ids.tolist()[0]


def test_sasrec_forward_and_scores_shape() -> None:
    model = SASRec(num_items=5, max_seq_len=3, embedding_dim=8, num_heads=2, num_layers=1)
    input_ids = torch.tensor([[0, 1, 2], [0, 3, 4]])

    user_repr = model(input_ids)
    scores = model.score_items(user_repr)

    assert user_repr.shape == (2, 8)
    assert scores.shape == (2, 6)


def test_sasrec_recommend_masks_padding_and_seen() -> None:
    model = SASRec(num_items=5, max_seq_len=3, embedding_dim=8, num_heads=2, num_layers=1)
    input_ids = torch.tensor([[0, 1, 2]])
    exclude_ids = torch.tensor([[1, 2]])

    _, item_ids = model.recommend(input_ids, top_k=3, exclude_ids=exclude_ids)

    assert 0 not in item_ids.tolist()[0]
    assert 1 not in item_ids.tolist()[0]
    assert 2 not in item_ids.tolist()[0]
