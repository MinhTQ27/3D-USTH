import pyvista as pv

point_id = 2500
p = pv.Plotter()

mesh = pv.read('D:\\Learn_and_Study\\USTH\\Bachelor\\3D_Project\\CG_dataset\\brick_part01.obj')


previous_value = 1  # Initialize previous value to 1 or any other starting value
mesh_actors = []  # Store mesh actors for easy removal

def create_mesh(value):
    global previous_value  # Use the global previous_value to track the last value

    # Round and convert the slider value to integer to ensure it always uses integer values
    value = int(round(value))

    # Only clear the meshes if the current value is smaller than the previous value
    if value < previous_value:
        # Remove the previous meshes by deleting the actors
        for actor in mesh_actors:
            p.remove_actor(actor)
        mesh_actors.clear()

    # Store the current value for the next time
    previous_value = value

    # Add the new meshes
    p.add_mesh(mesh, show_edges=True)
    red_point = p.add_mesh(mesh.points[point_id], color="red", point_size=10)
    mesh_actors.append(red_point)
    
    connected = mesh.point_neighbors_levels(point_id, n_levels=value)  # Get neighbors based on the slider value
    for i, ids in enumerate(connected, start=1):
        blue_point = p.add_mesh(mesh.points[ids], color="blue", point_size=10)
        mesh_actors.append(blue_point)
    
    return

# Set the slider range and value as integers and allow it to dynamically change
p.add_slider_widget(create_mesh, [1, 10], value=1, title='Level of neighborhood')
p.show(cpos="yx")
