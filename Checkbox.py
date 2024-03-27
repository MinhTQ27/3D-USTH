#Checkbox
import pyvista as pv

mesh = pv.read("C:/Users/84943/Downloads/part1_uv.ply")
pid=2500

p = pv.Plotter()
actor = p.add_mesh(mesh, show_edges=True, copy_mesh=True)
p.add_mesh(mesh, show_edges=True)
p.add_mesh(mesh.points[pid], color="red", point_size=10)

def toggle_vis(flag):
    res = 2
    connected = mesh.point_neighbors_levels(pid, n_levels=res)
    for i, ids in enumerate(connected, start=1):
        p.add_mesh(mesh.points[ids], color="blue", point_size=10)
    actor.SetVisibility(flag)
def toggle_vis1(flag):
    res = 5
    connected = mesh.point_neighbors_levels(pid, n_levels=res)
    for i, ids in enumerate(connected, start=1):
        p.add_mesh(mesh.points[ids], color="blue", point_size=10)
    actor.SetVisibility(flag)
def toggle_vis2(flag):
    res = 8
    connected = mesh.point_neighbors_levels(pid, n_levels=res)
    for i, ids in enumerate(connected, start=1):
        p.add_mesh(mesh.points[ids], color="blue", point_size=10)
    actor.SetVisibility(flag)

p.add_checkbox_button_widget(toggle_vis, value=True)
p.add_checkbox_button_widget(toggle_vis1, value=True, color_on="red", position=(10.0, 66))
p.add_checkbox_button_widget(toggle_vis2, value=True, color_on="green",  position=(10.0, 66+56))

p.show()
