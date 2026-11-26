"""Tests for the PCA of the 3D point cloud."""

import numpy as np
import pytest

from earth_similarity_explorer.cloud import (
    PCA,
    fit_pca,
    project_positions,
    quantize_positions,
)
from earth_similarity_explorer.settings import EMBEDDING_BANDS


# TESTS
def test_axes_follow_the_segments_with_the_most_pixels() -> None:
    """The first axis runs along the spread of the heavier segments and explains the most."""
    # Two segments of 100 pixels spread along A00, and two of 1 pixel spread twice as far
    # along A01. Unweighted, A01 would be the first axis.
    vectors: np.ndarray = np.zeros(shape=(4, len(EMBEDDING_BANDS)))
    vectors[:, :2] = [[0.5, 0], [-0.5, 0], [0, 1], [0, -1]]
    weights: np.ndarray = np.array(object=[100, 100, 1, 1])

    pca: PCA = fit_pca(vectors=vectors, weights=weights)

    assert abs(pca.axes[0, 0]) == pytest.approx(expected=1)
    # Weighted variances 200 × 0.5² along A00 and 2 × 1² along A01 give a share of 25/26
    assert pca.explained_variance[0] == pytest.approx(expected=25 / 26)


def test_global_axes_have_a_positive_largest_component() -> None:
    """Without reference axes, the largest component of each axis is positive."""
    rng: np.random.Generator = np.random.default_rng(seed=0)
    vectors: np.ndarray = rng.standard_normal(size=(50, len(EMBEDDING_BANDS)))

    pca: PCA = fit_pca(vectors=vectors, weights=np.ones(shape=50))

    # The component with the largest absolute value in each row (axis)
    largest: np.ndarray = pca.axes[np.arange(3), np.argmax(a=np.abs(pca.axes), axis=1)]
    assert (largest > 0).all()


def test_axes_point_the_same_way_as_the_reference_axes() -> None:
    """Each axis is flipped if it points against the reference axis of the same rank."""
    rng: np.random.Generator = np.random.default_rng(seed=0)
    vectors: np.ndarray = rng.standard_normal(size=(50, len(EMBEDDING_BANDS)))
    weights: np.ndarray = np.ones(shape=50)
    unaligned: PCA = fit_pca(vectors=vectors, weights=weights)
    # References that point against the first and third axis and along the second
    reference_axes: np.ndarray = unaligned.axes * np.array(object=[[-1], [1], [-1]])

    aligned: PCA = fit_pca(
        vectors=vectors, weights=weights, reference_axes=reference_axes
    )

    assert np.allclose(aligned.axes, reference_axes)


def test_positions_are_coordinates_along_the_axes_around_the_mean() -> None:
    """A segment's position is its offset from the mean, measured along each axis."""
    # The vectors of the first test, whose weighted mean is 0 and whose first two axes
    # are A00 and A01
    vectors: np.ndarray = np.zeros(shape=(4, len(EMBEDDING_BANDS)))
    vectors[:, :2] = [[0.5, 0], [-0.5, 0], [0, 1], [0, -1]]
    pca: PCA = fit_pca(vectors=vectors, weights=np.array(object=[100, 100, 1, 1]))

    positions: np.ndarray = project_positions(pca=pca, vectors=vectors)

    # The A00 and A01 values themselves, with PC3 at 0 since no segment varies along it
    expected: np.ndarray = np.array(
        object=[[0.5, 0, 0], [-0.5, 0, 0], [0, 1, 0], [0, -1, 0]]
    )
    assert np.allclose(positions, expected)


def test_quantized_axes_share_one_scale_centred_at_128() -> None:
    """The origin becomes 128, and an axis with half the spread fills half the bytes."""
    # PC1 spreads from −1 to 1, PC2 from −0.5 to 0.5, and PC3 not at all
    positions: np.ndarray = np.zeros(shape=(1000, 3))
    positions[:, 0] = np.linspace(start=-1, stop=1, num=1000)
    positions[:, 1] = np.linspace(start=-0.5, stop=0.5, num=1000)

    levels: np.ndarray = quantize_positions(positions=positions)

    assert (levels[:, 2] == 128).all()
    assert (levels[:, 0].min(), levels[:, 0].max()) == (1, 255)
    assert (levels[:, 1].min(), levels[:, 1].max()) == (64, 192)


def test_quantization_clips_outliers_instead_of_shrinking_the_cloud() -> None:
    """One far outlier lands on the edge byte, and the other segments keep their bytes."""
    positions: np.ndarray = np.zeros(shape=(1000, 3))
    positions[:, 0] = np.linspace(start=-1, stop=1, num=1000)
    with_outlier: np.ndarray = np.vstack(tup=[positions, [[100, 0, 0]]])

    levels: np.ndarray = quantize_positions(positions=with_outlier)

    assert levels[-1, 0] == 255
    # Without clipping, the outlier would set the scale and squeeze PC1 into 127 to 129
    assert np.array_equal(levels[:-1], quantize_positions(positions=positions))
