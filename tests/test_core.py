"""Fast unit tests (CPU, no downloads): `python -m pytest tests`."""

import numpy as np
import pytest
import torch

from eurofm import data
from eurofm.finetune import augment
from eurofm.vlm import parse_answer


@pytest.fixture(scope="module")
def ds():
    labels = np.repeat(np.arange(10), 30)
    test_idx, pool_idx = data.build_split(labels, test_per_class=10, seed=0)
    images = np.zeros((len(labels), 64, 64, 3), np.uint8)
    return data.EuroSAT(images, labels, test_idx, pool_idx, list(data.CLASS_NAMES))


def test_split_is_disjoint_and_stratified(ds):
    assert not set(ds.test_idx) & set(ds.pool_idx)
    assert len(ds.test_idx) + len(ds.pool_idx) == len(ds.labels)
    assert np.bincount(ds.labels[ds.test_idx]).tolist() == [10] * 10


def test_kshot_is_balanced_nested_and_from_pool(ds):
    one, five = data.sample_kshot(ds, 1, seed=0), data.sample_kshot(ds, 5, seed=0)
    assert np.bincount(ds.labels[five]).tolist() == [5] * 10
    assert set(one) <= set(five) <= set(ds.pool_idx)
    assert not np.array_equal(five, data.sample_kshot(ds, 5, seed=1))
    assert len(data.sample_kshot(ds, None, seed=0)) == len(ds.pool_idx)


def test_budgets():
    assert data.parse_budgets("1, 5,all") == [1, 5, None]
    assert data.budget_name(None) == "all"


def test_augment_preserves_pixels():
    x = torch.randint(0, 255, (16, 64, 64, 3), dtype=torch.uint8)
    y = augment(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    # flips and 90-degree rotations only permute pixels
    assert torch.equal(torch.sort(y.flatten(1), dim=1).values, torch.sort(x.flatten(1), dim=1).values)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Forest", "Forest"),
        ("SeaLake", "SeaLake"),
        ("sea or lake", "SeaLake"),
        ("The image shows a River.", "River"),
        ("<think>maybe forest</think>Highway", "Highway"),
        ("annual crop land", "AnnualCrop"),
        ("Permanent Crop", "PermanentCrop"),
        ("Herbaceous Vegetation", "HerbaceousVegetation"),
        ("a broad seasonal pasture", "Pasture"),  # 'road' / 'sea' must not match inside words
        ("Residential area", "Residential"),
        ("I don't know", None),
    ],
)
def test_parse_answer(text, expected):
    got = parse_answer(text)
    assert got == (-1 if expected is None else data.CLASS_NAMES.index(expected))
