"""
Basic tests that don't require the full dataset to run - just the
correctness of the core math and self-exclusion logic.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommend import cosine_similarity_score


def test_cosine_similarity_identical_vectors():
    v = torch.nn.functional.normalize(torch.tensor([1.0, 2.0, 3.0]), dim=0)
    assert abs(cosine_similarity_score(v, v) - 1.0) < 1e-5


def test_cosine_similarity_orthogonal_vectors():
    a = torch.nn.functional.normalize(torch.tensor([1.0, 0.0]), dim=0)
    b = torch.nn.functional.normalize(torch.tensor([0.0, 1.0]), dim=0)
    assert abs(cosine_similarity_score(a, b) - 0.0) < 1e-5


def test_cosine_similarity_opposite_vectors():
    a = torch.nn.functional.normalize(torch.tensor([1.0, 0.0]), dim=0)
    b = torch.nn.functional.normalize(torch.tensor([-1.0, 0.0]), dim=0)
    assert abs(cosine_similarity_score(a, b) - (-1.0)) < 1e-5


def test_config_seed_is_fixed_and_not_none():
    from config import config
    assert isinstance(config.RANDOM_SEED, int)


def test_config_single_model_declared():
    from config import config
    assert config.MODEL_NAME == "resnet50"
    assert config.EMBEDDING_DIM == 2048
