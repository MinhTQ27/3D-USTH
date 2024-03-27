#Neighborhood points
import pyvista as pv
import numpy as np

i = 1
while i <= 3:
    mesh = pv.read("C:/Users/84943/Downloads/part1_uv.ply")
    pid = 2500
    hihi = int(input())
    connected = mesh.point_neighbors_levels(2500, n_levels=hihi)

    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True)
    p.add_mesh(mesh.points[pid], color="red", point_size=10)
    for i, ids in enumerate(connected, start=1):
        p.add_mesh(mesh.points[ids], color="blue", point_size=10)

    p.show(cpos="yx")
    i+=1
