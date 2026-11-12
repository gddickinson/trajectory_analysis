"""Small, dependency-light track geometry helpers shared by features.py and advanced_metrics.py.

fractal_dimension: the fractal dimension of the original analysis scripts,
    log(N) / log(N * d_max / L), with N the number of points, L the path length and d_max the largest distance
    between any two points (found on the convex hull). NaN for fewer than 3 points, all points on one line through
    the first point, or a degenerate hull. Reproduces the fracDimension column of the example results file.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def fractal_dimension(positions: np.ndarray) -> float:
    """Fractal dimension of an (N, 2) track as in the original scripts (see module docstring)."""
    try:
        if len(positions) < 3:
            return np.nan

        x0, y0 = positions[0]
        points = [(x, y) for x, y in positions if (x != x0) or (y != y0)]
        if len(points) < 2:
            return np.nan

        slopes = [((y - y0) / (x - x0)) if (x != x0) else None for x, y in points]
        if all(s == slopes[0] for s in slopes):
            return np.nan  # collinear points

        total_path_length = np.sum(np.sqrt(np.sum((positions[1:] - positions[:-1]) ** 2, axis=1)))

        hull = ConvexHull(positions)
        candidates = positions[hull.vertices]
        largest_distance = np.max(cdist(candidates, candidates))

        step_count = len(positions)
        if largest_distance > 0 and total_path_length > 0:
            return math.log(step_count) / math.log(step_count * largest_distance / total_path_length)
        return np.nan
    except Exception:
        return np.nan
