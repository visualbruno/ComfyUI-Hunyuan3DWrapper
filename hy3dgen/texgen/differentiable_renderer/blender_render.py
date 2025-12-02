import bpy
import sys
import argparse
import math
import os

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def configure_gpu(rendering):    
    if rendering == 'GPU':    
        bpy.context.scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences

        for compute_device_type in ('OPTIX', 'CUDA', 'HIP', 'METAL'):
            try:
                cprefs.compute_device_type = compute_device_type
                cprefs.get_devices()
                devices = cprefs.devices
                if devices:
                    print(f"Blender: Found {compute_device_type} devices:")
                    for device in devices:
                        device.use = True
                        print(f"    - Activated: {device.name}")
                    return
            except Exception as e:
                continue
        print("Blender: No GPU found, falling back to CPU.")
        bpy.context.scene.cycles.device = 'CPU'
    else:
        bpy.context.scene.cycles.device = 'CPU'

def set_camera(elev, azim, distance, scale, clip_start=0.1, clip_end=100):
    # Location calculation
    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim - 90)
    
    x = distance * math.cos(elev_rad) * math.cos(azim_rad)
    y = distance * math.cos(elev_rad) * math.sin(azim_rad)
    z = distance * math.sin(elev_rad)
    
    camera_data = bpy.data.cameras.new(name='Camera')
    
    # Orthographic settings
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = scale
    camera_data.clip_start = clip_start
    camera_data.clip_end = clip_end

    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    
    camera_object.location = (x, y, z)
    
    # Point camera at origin using a Track To constraint
    bpy.ops.object.empty_add(location=(0, 0, 0))
    target = bpy.context.active_object
    
    constraint = camera_object.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

def setup_lighting():
    # Ensure a World data block exists (fix for factory settings)
    if bpy.context.scene.world is None:
        new_world = bpy.data.worlds.new("New_Render_World")
        bpy.context.scene.world = new_world
    
    world = bpy.context.scene.world
    world.use_nodes = True
    
    # Get or create the Background node
    if 'Background' not in world.node_tree.nodes:
        world.node_tree.nodes.new('ShaderNodeBackground')
        output_node = world.node_tree.nodes['World Output']
        bg_node = world.node_tree.nodes['Background']
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    bg_node = world.node_tree.nodes['Background']
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs['Strength'].default_value = 1.0
    
    bpy.context.scene.view_settings.view_transform = 'Standard'
    
    bpy.context.scene.cycles.max_bounces = 8
    bpy.context.scene.cycles.diffuse_bounces = 4 # How many times light reflects off diffuse surfaces

def auto_center_and_scale(obj, norm_size):
    """
    Centers the mesh object's geometry at the origin and scales it such that 
    (Max Radius * 2.0) equals norm_size.
    """
    if obj.type != 'MESH':
        print(f"Object {obj.name} is not a mesh. Skipping transformation.")
        return

    # Ensure we are in Object Mode before modifying geometry
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Get the bounding box coordinates (min/max for X, Y, Z)
    # The bounding box is an 8-tuple of (x,y,z) coordinates.
    # The Bounding Box is in local coordinates when the object's scale is (1,1,1)
    bbox = obj.bound_box

    # Find min and max coordinates
    min_x = min(v[0] for v in bbox)
    max_x = max(v[0] for v in bbox)
    min_y = min(v[1] for v in bbox)
    max_y = max(v[1] for v in bbox)
    min_z = min(v[2] for v in bbox)
    max_z = max(v[2] for v in bbox)

    # Calculate Center (equivalent to your (max_bb + min_bb) / 2)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    center = (center_x, center_y, center_z)

    # 1. Translate Geometry (Equivalent to vtx_pos = (vtx_pos - center))
    # This moves the geometry relative to the object's local origin.
    # The context must be set correctly for bpy.ops.transform.translate to work.
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Enter Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Translate vertices by -center vector
    bpy.ops.transform.translate(value=(-center_x, -center_y, -center_z))
    
    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # The object's geometry is now centered on the local origin (0,0,0).
    # We ensure the object's world location is also (0,0,0)
    obj.location = (0, 0, 0)
    
    # 2. Calculate Maximum Radius (for scaling)
    current_max_radius_sq = 0.0
    mesh = obj.data
    
    # Iterate over vertices to find the farthest one from the origin
    # The vertex coordinates are now relative to the center
    for vertex in mesh.vertices:
        if vertex.co.length_squared > current_max_radius_sq:
            current_max_radius_sq = vertex.co.length_squared
    
    current_max_radius = math.sqrt(current_max_radius_sq)

    if current_max_radius < 1e-6:
        print("Skipping scaling: mesh has negligible size.")
        return

    # 3. Calculate and Apply Scale
    # User's scaling basis: scale_user = current_max_radius * 2.0
    scale_user = current_max_radius * 2.0
    
    # Target scale factor: (scale_factor / scale_user)
    scale_factor_needed = norm_size / scale_user
    
    # Apply scale to the object
    obj.scale = (scale_factor_needed, scale_factor_needed, scale_factor_needed)
    
    # 4. Apply transformation to bake scale into geometry (optional but good practice)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    print(f"Mesh '{obj.name}' centered. Scaled by {scale_factor_needed:.4f}.")
    print(f"Final normalized size (Max Radius * 2.0) is approx {norm_size:.4f}.")

def import_mesh(mesh_path):
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        sys.exit(1)

    if mesh_path.endswith('.obj'):
        if hasattr(bpy.ops.wm, 'obj_import'):
            bpy.ops.wm.obj_import(filepath=mesh_path)
        else:
            bpy.ops.import_scene.obj(filepath=mesh_path)
    elif mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else:
        print(f"Unsupported mesh format: {mesh_path}")
        sys.exit(1)
    
    # Return the first imported mesh object
    return bpy.context.view_layer.objects.active

def render_scene(output_path, resolution, rendering):
    abs_output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(abs_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    bpy.context.scene.render.engine = 'CYCLES'
    configure_gpu(rendering)
    
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = abs_output_path
    bpy.context.scene.render.use_file_extension = False
    
    bpy.context.scene.cycles.samples = 512
    bpy.context.scene.cycles.preview_samples = 32

    bpy.context.scene.cycles.use_denoising = True
    
    print(f"Rendering to: {abs_output_path}")
    bpy.ops.render.render(write_still=True)

def main():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--elev', type=float, required=True)
    parser.add_argument('--azim', type=float, required=True)
    parser.add_argument('--distance', type=float, default=1.45)
    parser.add_argument('--scale', type=float, default=2.0, help="Orthographic scale (field of view size)")
    # New argument for normalization size
    parser.add_argument('--norm_size', type=float, default=1.15, help="Target size for Max Radius * 2.0")
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--rendering', default="GPU")
    
    args = parser.parse_args(argv)
    
    reset_scene()
    
    # Import and get the mesh object
    mesh_obj = import_mesh(args.mesh)
    
    # Auto-center and scale the mesh
    if mesh_obj:
        auto_center_and_scale(mesh_obj, args.norm_size)
    
    # Set up the camera and lighting
    set_camera(args.elev, args.azim, args.distance, args.scale)
    setup_lighting()
    
    render_scene(args.output, args.resolution, args.rendering)

if __name__ == "__main__":
    main()