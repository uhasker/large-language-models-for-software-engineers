#!/usr/bin/env python3
"""
IVFFlat (Inverted File with Flat) illustration

- Train coarse centroids with a tiny K-Means.
- Partition space into cells (visualized via nearest-centroid on a grid).
- At query time, probe the k nearest cells and do exact search only inside them.
- Compare with brute-force exact nearest neighbors.
- Save two figures illustrating the partition and the probed cells.

This script is self-contained and uses only NumPy + Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

rng = np.random.default_rng(42)

N_POINTS = 1200
N_CENTROIDS = 15
K_PROBES = 3
N_NEIGHBORS = 25
N_KMEANS_ITERS = 12
PAD = 0.5
GRID_RES = 400

centers_true = np.array([[-3, -2], [0.5, 1.0], [3.0, -1.0], [2.5, 3.5]])
scales = np.array([1.0, 0.8, 0.9, 0.6])

points = []
for c, s in zip(centers_true, scales):
    m = N_POINTS // len(centers_true)
    pts = c + rng.normal(size=(m, 2)) * s
    points.append(pts)
X = np.vstack(points)
N_POINTS = X.shape[0]

query = rng.normal(size=(1, 2)) * 0.7 + np.array([[0.0, 0.5]])

def kmeans_lloyd(data: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    init_idx = rng.choice(data.shape[0], size=k, replace=False)
    centroids = data[init_idx].copy()
    for _ in range(iters):
        d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = data[mask].mean(axis=0)
            else:
                centroids[j] = data[rng.integers(0, data.shape[0])]
    d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    return centroids, labels

centroids, labels = kmeans_lloyd(X, N_CENTROIDS, N_KMEANS_ITERS, rng)

def ivfflat_search(data, centroids, labels, query, k_probes, n_neighbors):
    d2_cent = ((centroids[None, :, :] - query[:, None, :]) ** 2).sum(axis=2)[0]
    probe_order = np.argsort(d2_cent)
    probe_ids = probe_order[:k_probes]
    candidate_mask = np.isin(labels, probe_ids)
    candidates = np.where(candidate_mask)[0]
    d2 = ((data[candidates] - query[0]) ** 2).sum(axis=1)
    nn_local_idx = np.argsort(d2)[:n_neighbors]
    nn_global_idx = candidates[nn_local_idx]
    nn_dists = np.sqrt(d2[nn_local_idx])
    d2_all = ((data - query[0]) ** 2).sum(axis=1)
    bf_order = np.argsort(d2_all)[:n_neighbors]
    bf_dists = np.sqrt(d2_all[bf_order])
    return nn_global_idx, nn_dists, bf_order, bf_dists, probe_ids

# background grid (nearest-centroid cells)
xmin, ymin = X.min(axis=0) - PAD
xmax, ymax = X.max(axis=0) + PAD
xs = np.linspace(xmin, xmax, GRID_RES)
ys = np.linspace(ymin, ymax, GRID_RES)
xx, yy = np.meshgrid(xs, ys)
grid = np.stack([xx, yy], axis=-1)
d2_grid = ((grid[:, :, None, :] - centroids[None, None, :, :]) ** 2).sum(axis=3)
grid_labels = d2_grid.argmin(axis=2)

# Plot 1
fig1 = plt.figure(figsize=(12, 8))
plt.pcolormesh(xx, yy, grid_labels, shading="auto", cmap="Blues", alpha=0.7)
plt.scatter(X[:, 0], X[:, 1], s=6, c="darkgreen", alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", linewidths=2, c="red")
plt.scatter(query[:, 0], query[:, 1], s=200, marker="*", c="orange")
plt.title("IVFFlat: space partitioned into centroid cells\n(centroids - red ×, query - orange *)")
plt.xlabel("x"); plt.ylabel("y")
fig1.tight_layout()
fig1.savefig("ivfflat_cells.png", dpi=150)

# Search + Plot 2
ivf_idx, ivf_dists, bf_idx, bf_dists, probe_ids = ivfflat_search(X, centroids, labels, query, K_PROBES, N_NEIGHBORS)
selected = np.isin(grid_labels, probe_ids)
grid_selected = np.where(selected, grid_labels, -1)

fig2 = plt.figure(figsize=(12, 8))
plt.pcolormesh(xx, yy, grid_selected, shading="auto", cmap="Blues", alpha=0.7)
plt.scatter(X[:, 0], X[:, 1], s=6, c="darkgreen", alpha=0.8)
plt.scatter(X[np.isin(labels, probe_ids), 0], X[np.isin(labels, probe_ids), 1], s=6, c="purple", alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", linewidths=2, c="red")
plt.scatter(X[ivf_idx, 0], X[ivf_idx, 1], s=80, marker="^", c="magenta")
plt.scatter(query[:, 0], query[:, 1], s=220, marker="*", c="orange")
plt.title(f"IVFFlat search: probing {K_PROBES} nearest cells, returning {N_NEIGHBORS} neighbors\n"
          "candidates (purple •), results (magenta ▲), centroids (red ×), query (orange *)")
plt.xlabel("x"); plt.ylabel("y")
fig2.tight_layout()
fig2.savefig("ivfflat_search.png", dpi=150)

print("Saved: ivfflat_cells.png, ivfflat_search.png")