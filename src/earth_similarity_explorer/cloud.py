"""The 3D point cloud, which places segments by a weighted PCA of their vectors."""

from dataclasses import dataclass

import numpy as np

from earth_similarity_explorer.settings import (
    CLOUD_FIT_SAMPLE_SIZE,
    CLOUD_FIT_SEED,
    POSITION_BYTE_CENTRE,
    POSITION_BYTE_HALF_RANGE,
    POSITION_CLIP_PERCENTILE,
)


@dataclass(frozen=True)
class PCA:
    """Three PCA axes for a set of segment vectors."""

    # Weighted mean vector (the centre of the cloud)
    mean: np.ndarray
    # The axes as unit vectors, one per row (3 × 64)
    axes: np.ndarray
    # Share of the total variance along each axis (3)
    explained_variance: np.ndarray


def fit_pca(
    vectors: np.ndarray, weights: np.ndarray, reference_axes: np.ndarray | None = None
) -> PCA:
    """Fit the three principal axes of segment vectors, weighted by pixel count.

    If reference_axes (3 × 64, the global axes) is supplied each axis is flipped
    to point the same way as the reference axis of the same rank.
    Without them each axis is flipped so that its largest component is positive.
    """
    # Calculate the weighted mean (by segment size) for each band (64)
    # The mean vector is the centre of the cloud
    mean: np.ndarray = np.average(a=vectors, axis=0, weights=weights)
    # Calculate the weighted covariance (by segment size) between the bands (64 × 64)
    covariance: np.ndarray = np.cov(
        m=vectors, rowvar=False, aweights=weights, bias=True
    )
    # Calculate the eigenvectors and their variances
    variances, eigenvectors = np.linalg.eigh(a=covariance)
    # Keep the three eigenvectors with the highest variance
    axes: np.ndarray = eigenvectors[:, :-4:-1].T
    # Calculate the explained variance captured by the first three principal components
    explained_variance: np.ndarray = variances[:-4:-1] / variances.sum()

    # Without reference axes, the largest value of each axis sets its positive direction
    if reference_axes is None:
        signed_by: np.ndarray = axes[np.arange(3), np.argmax(a=np.abs(axes), axis=1)]
    # With reference axes, follow their orientation, so the clouds of different
    # selections don't appear mirrored
    else:
        signed_by = np.vecdot(axes, reference_axes)
    # Multiply each axis whose chosen value is negative by −1 to flip it
    signs: np.ndarray = np.where(signed_by < 0, -1.0, 1.0)
    return PCA(
        mean=mean, axes=axes * signs[:, None], explained_variance=explained_variance
    )


def project_positions(pca: PCA, vectors: np.ndarray) -> np.ndarray:
    """Place segment vectors in the cloud as coordinates along the PCA axes (N × 3)."""
    # Subtract the mean so the centre of the cloud becomes the origin (0, 0, 0)
    # and take the dot product with each axis (one column per axis)
    return (vectors - pca.mean) @ pca.axes.T


def quantize_positions(positions: np.ndarray) -> np.ndarray:
    """Turn cloud positions (N × 3) into one byte per uint8 axis centred at 128."""
    # Calculate one clipping range for all three axes, so PC3 keeps its smaller spread
    clip_range: float = np.percentile(a=np.abs(positions), q=POSITION_CLIP_PERCENTILE)
    # Clip the outliers beyond the range to its edge
    clipped: np.ndarray = np.clip(a=positions, a_min=-clip_range, a_max=clip_range)
    # Scale −range…range to 1…255 with the origin at 128
    levels: np.ndarray = np.round(
        a=POSITION_BYTE_CENTRE + clipped / clip_range * POSITION_BYTE_HALF_RANGE
    )
    return levels.astype(dtype=np.uint8)


def build_cloud(
    vectors: np.ndarray, weights: np.ndarray, reference_axes: np.ndarray | None = None
) -> tuple[np.ndarray, PCA]:
    """Fit a PCA to a sample of the segments and generate 3D cloud coordinates in uint8."""
    # Draw at most CLOUD_FIT_SAMPLE_SIZE segments to fit on (fixed seed)
    rng: np.random.Generator = np.random.default_rng(seed=CLOUD_FIT_SEED)
    sample: np.ndarray = rng.choice(
        a=len(vectors), size=min(len(vectors), CLOUD_FIT_SAMPLE_SIZE), replace=False
    )
    pca: PCA = fit_pca(
        vectors=vectors[sample], weights=weights[sample], reference_axes=reference_axes
    )
    # Place every segment (not only the sample) and turn the positions into bytes
    positions: np.ndarray = project_positions(pca=pca, vectors=vectors)
    return quantize_positions(positions=positions), pca
