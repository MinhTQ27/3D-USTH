#Slide bar
import pyvista as pv
import numpy as np

mesh = pv.read("C:/Users/84943/Downloads/part1_uv.ply")
pid = 2500
p = pv.Plotter()

def create_mesh(value):
    p.add_mesh(mesh, show_edges=True)
    p.add_mesh(mesh.points[pid], color="red", point_size=10)
    res = int(value)
    connected = mesh.point_neighbors_levels(pid, n_levels=res)
    for i, ids in enumerate(connected, start=1):
        p.add_mesh(mesh.points[ids], color="blue", point_size=10)
    return

p.add_slider_widget(create_mesh, [1, 10], value=1, title='Level of neighborhood')
p.show(cpos="yx")
