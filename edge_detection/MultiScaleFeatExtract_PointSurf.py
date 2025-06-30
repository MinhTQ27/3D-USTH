#!/home/minhtrinh/my_project/venv/project3D/bin/python

# This code is implemented by ChatGPT. Please check again the algorithm ----------

import pyvista as pv
import numpy as np
from sklearn.neighbors import NearestNeighbors

# === Load mesh ===
mesh = pv.read("../CG_dataset/brick_part01.obj")
points = mesh.points
n_points = len(points)

# === Parameters ===
k_scales = [10, 20, 30, 40, 50]  # Multi-scale neighborhood sizes
sigma_thresh = 0.01              # Surface variation threshold (tune this!)
feature_weights = np.zeros(n_points)  # To store persistence count per point

# === Build kNN search structure ===
nn_model = NearestNeighbors(n_neighbors=max(k_scales), algorithm='auto')
nn_model.fit(points)

# === For each scale, compute surface variation ===
for k in k_scales:
    distances, indices = nn_model.kneighbors(points, n_neighbors=k)

    for i in range(n_points):
        neighbors = points[indices[i]]
        centroid = np.mean(neighbors, axis=0)
        diffs = neighbors - centroid
        cov = np.dot(diffs.T, diffs) / k

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)  # Sorted ascending
        lambda0, lambda1, lambda2 = eigenvalues
        variation = lambda0 / (lambda0 + lambda1 + lambda2 + 1e-10)  # Avoid zero division

        # Classification: surface variation exceeds threshold?
        if variation > sigma_thresh:
            feature_weights[i] += 1

# === Normalize weights ===
feature_weights /= len(k_scales)

# === Add to mesh and visualize ===
mesh["feature_score"] = feature_weights
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="feature_score", cmap="coolwarm", point_size=5)
plotter.show()
