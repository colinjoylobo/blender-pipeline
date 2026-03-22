"""
Blender Texture Baker - High Poly to Low Poly
================================================
Bakes textures from a high-poly model onto a low-poly model.

Supported formats: FBX, OBJ, GLB, GLTF, PLY, STL, DAE, ABC

Usage (command line):
    blender --background --python bake_textures.py -- \
        --highpoly /path/to/highpoly.fbx \
        --lowpoly /path/to/lowpoly.fbx \
        --output /path/to/output_dir \
        --resolution 4096 \
        --cage-extrusion 0.05 \
        --maps normal diffuse ao roughness metallic

Usage (inside Blender scripting tab):
    Set the variables in the CONFIG section below, then run.
"""

import bpy
import os
import sys
import math
import argparse
from pathlib import Path


# ============================================================
# CONFIG - Edit these if running inside Blender's scripting tab
# ============================================================
CONFIG = {
    "highpoly": "",       # Path to high-poly model (leave empty if using CLI args)
    "lowpoly": "",        # Path to low-poly model
    "output": "",         # Output directory for baked textures
    "resolution": 4096,   # Texture resolution (e.g., 1024, 2048, 4096)
    "cage_extrusion": 0.05,  # Ray distance for baking (increase if artifacts appear)
    "max_ray_distance": 0.0,  # 0 = auto
    "maps": ["normal", "diffuse", "ao", "roughness", "metallic"],
    "format": "PNG",      # Output format: PNG, EXIF, TIFF, OPEN_EXR
    "samples": 128,       # Render samples for baking (higher = cleaner but slower)
    "margin": 16,         # Texture margin in pixels (prevents UV seam bleeding)
    "export_lowpoly": True,  # Export the low-poly with baked materials applied
}
# ============================================================


SUPPORTED_EXTENSIONS = {
    ".fbx", ".obj", ".glb", ".gltf", ".ply", ".stl", ".dae", ".abc", ".blend"
}

MAP_CONFIGS = {
    "normal": {
        "type": "NORMAL",
        "color_space": "Non-Color",
        "suffix": "_normal",
        "use_alpha": False,
    },
    "diffuse": {
        "type": "DIFFUSE",
        "color_space": "sRGB",
        "suffix": "_diffuse",
        "use_alpha": False,
    },
    "ao": {
        "type": "AO",
        "color_space": "Non-Color",
        "suffix": "_ao",
        "use_alpha": False,
    },
    "roughness": {
        "type": "ROUGHNESS",
        "color_space": "Non-Color",
        "suffix": "_roughness",
        "use_alpha": False,
    },
    "metallic": {
        "type": "EMIT",  # Metallic is baked via emission trick
        "color_space": "Non-Color",
        "suffix": "_metallic",
        "use_alpha": False,
    },
    "emit": {
        "type": "EMIT",
        "color_space": "sRGB",
        "suffix": "_emission",
        "use_alpha": False,
    },
    "combined": {
        "type": "COMBINED",
        "color_space": "sRGB",
        "suffix": "_combined",
        "use_alpha": False,
    },
}


def parse_args():
    """Parse command line arguments when run via blender --python."""
    if "--" not in sys.argv:
        return None

    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description="Bake textures from high-poly to low-poly")
    parser.add_argument("--highpoly", required=True, help="Path to high-poly model")
    parser.add_argument("--lowpoly", required=True, help="Path to low-poly model")
    parser.add_argument("--output", default="", help="Output directory (default: next to lowpoly)")
    parser.add_argument("--resolution", type=int, default=4096, help="Texture resolution")
    parser.add_argument("--cage-extrusion", type=float, default=0.05, help="Cage extrusion distance")
    parser.add_argument("--max-ray-distance", type=float, default=0.0, help="Max ray distance (0=auto)")
    parser.add_argument("--samples", type=int, default=128, help="Bake samples")
    parser.add_argument("--margin", type=int, default=16, help="UV margin in pixels")
    parser.add_argument("--maps", nargs="+", default=["normal", "diffuse", "ao", "roughness", "metallic"],
                        choices=list(MAP_CONFIGS.keys()), help="Maps to bake")
    parser.add_argument("--format", default="PNG", choices=["PNG", "TIFF", "OPEN_EXR"],
                        help="Output image format")
    parser.add_argument("--no-export", action="store_true", help="Skip exporting low-poly with materials")
    return parser.parse_args(argv)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def import_model(filepath):
    """Import a 3D model file. Returns list of imported mesh objects."""
    filepath = os.path.abspath(filepath)
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Track objects before import
    existing = set(obj.name for obj in bpy.data.objects)

    print(f"  Importing: {filepath}")

    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=filepath)
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=filepath)
    elif ext == ".abc":
        bpy.ops.wm.alembic_import(filepath=filepath)
    elif ext == ".blend":
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)

    # Collect newly imported objects (mesh only)
    new_objects = [obj for obj in bpy.data.objects
                   if obj.name not in existing and obj.type == 'MESH']

    if not new_objects:
        raise RuntimeError(f"No mesh objects imported from: {filepath}")

    print(f"  Imported {len(new_objects)} mesh object(s): {[o.name for o in new_objects]}")
    return new_objects


def join_objects(objects, name="Joined"):
    """Join multiple mesh objects into one. Returns the joined object."""
    if len(objects) == 1:
        objects[0].name = name
        return objects[0]

    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()

    result = bpy.context.active_object
    result.name = name
    return result


def ensure_uv_unwrap(obj):
    """Ensure the object has UV maps. Smart UV project if none exist."""
    if not obj.data.uv_layers:
        print(f"  No UVs found on '{obj.name}', running Smart UV Project...")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
        print("  Smart UV Project complete.")
    else:
        print(f"  UVs found on '{obj.name}': {[uv.name for uv in obj.data.uv_layers]}")


def auto_cage_extrusion(highpoly_obj, lowpoly_obj):
    """Estimate a good cage extrusion based on bounding box differences."""
    hp_dims = highpoly_obj.dimensions
    lp_dims = lowpoly_obj.dimensions
    avg_diff = sum(abs(hp_dims[i] - lp_dims[i]) for i in range(3)) / 3.0
    max_dim = max(max(hp_dims), max(lp_dims))

    # Use a fraction of the max dimension, at least the average difference
    extrusion = max(avg_diff * 1.5, max_dim * 0.02)
    print(f"  Auto cage extrusion: {extrusion:.4f}")
    return extrusion


def setup_cycles(samples):
    """Configure Cycles renderer for baking."""
    bpy.context.scene.render.engine = 'CYCLES'

    # Prefer GPU if available
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        cprefs = prefs.preferences
        # Try Metal (macOS), then CUDA, then OptiX, fallback to CPU
        for compute_type in ['METAL', 'OPTIX', 'CUDA', 'NONE']:
            try:
                cprefs.compute_device_type = compute_type
                cprefs.get_devices()
                if compute_type != 'NONE':
                    for device in cprefs.devices:
                        device.use = True
                    bpy.context.scene.cycles.device = 'GPU'
                    print(f"  Using GPU compute: {compute_type}")
                    break
            except Exception:
                continue
        else:
            bpy.context.scene.cycles.device = 'CPU'
            print("  Using CPU compute")

    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.use_denoising = False


def create_bake_image(name, resolution, color_space="Non-Color"):
    """Create a new image for baking."""
    img = bpy.data.images.new(name, width=resolution, height=resolution, alpha=False)
    img.colorspace_settings.name = color_space
    # Fill with appropriate default
    if "normal" in name.lower():
        img.generated_color = (0.5, 0.5, 1.0, 1.0)  # Flat normal
    else:
        img.generated_color = (0.0, 0.0, 0.0, 1.0)
    return img


def setup_bake_material(obj, bake_image):
    """Set up a material with an image texture node for baking on the target object.
    Returns the image node so it can be selected for baking."""
    # Ensure the object has a material
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="BakeMaterial")
        mat.use_nodes = True
        obj.data.materials.append(mat)

    mat = obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Create or find an image texture node for baking
    bake_node = nodes.new(type='ShaderNodeTexImage')
    bake_node.name = "BakeTarget"
    bake_node.image = bake_image
    bake_node.location = (-400, 300)

    # Make it the active/selected node (required for baking)
    for node in nodes:
        node.select = False
    bake_node.select = True
    nodes.active = bake_node

    return bake_node


def prepare_metallic_emission(highpoly_obj):
    """Temporarily reroute metallic to emission for baking metallic maps."""
    modified_materials = []

    for mat_slot in highpoly_obj.material_slots:
        mat = mat_slot.material
        if not mat or not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        principled = None

        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                break

        if not principled:
            continue

        # Find metallic input
        metallic_input = principled.inputs.get("Metallic")
        if not metallic_input:
            continue

        # Create an emission shader to output metallic as color
        emit_node = nodes.new(type='ShaderNodeEmission')
        emit_node.name = "_TempMetallicEmit"
        emit_node.location = (principled.location.x + 200, principled.location.y - 200)

        # Find output node
        output_node = None
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node
                break

        if not output_node:
            continue

        # Store original connection
        original_link = None
        for link in links:
            if link.to_node == output_node and link.to_socket.name == "Surface":
                original_link = (link.from_node.name, link.from_socket.name)
                break

        # If metallic has a connected texture, route it through emission
        if metallic_input.is_linked:
            source_socket = metallic_input.links[0].from_socket
            links.new(source_socket, emit_node.inputs["Color"])
        else:
            # Use the default value as a solid color
            val = metallic_input.default_value
            emit_node.inputs["Color"].default_value = (val, val, val, 1.0)

        emit_node.inputs["Strength"].default_value = 1.0
        links.new(emit_node.outputs["Emission"], output_node.inputs["Surface"])

        modified_materials.append({
            "material": mat,
            "emit_node_name": emit_node.name,
            "original_link": original_link,
            "output_node_name": output_node.name,
        })

    return modified_materials


def restore_metallic_emission(modified_materials):
    """Restore materials after metallic baking."""
    for info in modified_materials:
        mat = info["material"]
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Remove temp emission node
        emit_node = nodes.get(info["emit_node_name"])
        if emit_node:
            nodes.remove(emit_node)

        # Restore original link
        if info["original_link"]:
            src_node = nodes.get(info["original_link"][0])
            out_node = nodes.get(info["output_node_name"])
            if src_node and out_node:
                src_socket = src_node.outputs.get(info["original_link"][1])
                if src_socket:
                    links.new(src_socket, out_node.inputs["Surface"])


def bake_map(highpoly_obj, lowpoly_obj, map_name, config, resolution, cage_extrusion,
             max_ray_distance, margin, output_dir, img_format):
    """Bake a single texture map."""
    print(f"\n{'='*50}")
    print(f"  Baking: {map_name.upper()}")
    print(f"{'='*50}")

    base_name = lowpoly_obj.name.replace(" ", "_")
    img_name = f"{base_name}{config['suffix']}"

    # Create image
    bake_image = create_bake_image(img_name, resolution, config["color_space"])

    # Set up bake target on low-poly material
    bake_node = setup_bake_material(lowpoly_obj, bake_image)

    # Handle metallic special case
    modified_mats = []
    if map_name == "metallic":
        modified_mats = prepare_metallic_emission(highpoly_obj)

    # Select high-poly, then low-poly (active)
    bpy.ops.object.select_all(action='DESELECT')
    highpoly_obj.select_set(True)
    lowpoly_obj.select_set(True)
    bpy.context.view_layer.objects.active = lowpoly_obj

    # Make sure the bake node is active on every material slot
    for mat_slot in lowpoly_obj.material_slots:
        mat = mat_slot.material
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                node.select = False
            target = mat.node_tree.nodes.get("BakeTarget")
            if target:
                target.select = True
                mat.node_tree.nodes.active = target

    # Configure bake settings
    bake_settings = bpy.context.scene.render.bake
    bake_settings.use_selected_to_active = True
    bake_settings.cage_extrusion = cage_extrusion
    if max_ray_distance > 0:
        bake_settings.max_ray_distance = max_ray_distance
    bake_settings.margin = margin
    bake_settings.margin_type = 'EXTEND'
    bake_settings.use_cage = False

    # For diffuse, only bake color (no direct/indirect lighting)
    if config["type"] == "DIFFUSE":
        bake_settings.use_pass_direct = False
        bake_settings.use_pass_indirect = False
        bake_settings.use_pass_color = True

    # Bake
    try:
        bpy.ops.object.bake(type=config["type"])
    except RuntimeError as e:
        print(f"  ERROR baking {map_name}: {e}")
        if modified_mats:
            restore_metallic_emission(modified_mats)
        return None

    # Restore metallic materials
    if modified_mats:
        restore_metallic_emission(modified_mats)

    # Save image
    ext_map = {"PNG": ".png", "TIFF": ".tiff", "OPEN_EXR": ".exr"}
    ext = ext_map.get(img_format, ".png")
    output_path = os.path.join(output_dir, f"{img_name}{ext}")

    bake_image.filepath_raw = output_path
    bake_image.file_format = img_format
    # NOTE: Do NOT re-set colorspace_settings here — it causes generated-type
    # images to regenerate from generated_color, wiping baked pixel data.
    # Colorspace is already set correctly in create_bake_image().
    bake_image.save()

    print(f"  Saved: {output_path}")
    return output_path


def assign_baked_materials(lowpoly_obj, baked_maps, output_dir):
    """Create a proper material on the low-poly using baked textures."""
    mat = bpy.data.materials.new(name=f"{lowpoly_obj.name}_Baked")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create Principled BSDF
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    x_offset = -600
    y_offset = 300

    for map_name, filepath in baked_maps.items():
        if filepath is None:
            continue

        img = bpy.data.images.load(filepath)
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.image = img
        tex_node.location = (x_offset, y_offset)

        config = MAP_CONFIGS[map_name]
        tex_node.image.colorspace_settings.name = config["color_space"]

        if map_name == "diffuse":
            links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])
        elif map_name == "normal":
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (x_offset + 300, y_offset)
            links.new(tex_node.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
        elif map_name == "roughness":
            links.new(tex_node.outputs["Color"], principled.inputs["Roughness"])
        elif map_name == "metallic":
            links.new(tex_node.outputs["Color"], principled.inputs["Metallic"])
        elif map_name == "ao":
            # AO is typically multiplied with diffuse, add a mix node
            pass  # AO saved as separate map for compositing

        y_offset -= 300

    # Assign material
    lowpoly_obj.data.materials.clear()
    lowpoly_obj.data.materials.append(mat)
    print(f"\n  Material '{mat.name}' assigned to '{lowpoly_obj.name}'")


def export_lowpoly(lowpoly_obj, output_dir, original_lowpoly_path):
    """Export the low-poly model with baked materials."""
    base_name = Path(original_lowpoly_path).stem
    ext = Path(original_lowpoly_path).suffix.lower()

    bpy.ops.object.select_all(action='DESELECT')
    lowpoly_obj.select_set(True)
    bpy.context.view_layer.objects.active = lowpoly_obj

    # Export as FBX by default, or match original format
    if ext in (".glb", ".gltf"):
        out_path = os.path.join(output_dir, f"{base_name}_baked.glb")
        bpy.ops.export_scene.gltf(
            filepath=out_path,
            use_selection=True,
            export_format='GLB',
            export_materials='EXPORT',
        )
    elif ext == ".obj":
        out_path = os.path.join(output_dir, f"{base_name}_baked.obj")
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            export_materials=True,
        )
    else:
        out_path = os.path.join(output_dir, f"{base_name}_baked.fbx")
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=True,
            path_mode='COPY',
            embed_textures=True,
        )

    print(f"\n  Exported low-poly with baked textures: {out_path}")
    return out_path


def main():
    print("\n" + "=" * 60)
    print("  TEXTURE BAKER - High Poly to Low Poly")
    print("=" * 60)

    # Get configuration from CLI args or CONFIG dict
    args = parse_args()
    if args:
        highpoly_path = args.highpoly
        lowpoly_path = args.lowpoly
        output_dir = args.output or os.path.join(os.path.dirname(args.lowpoly), "baked_textures")
        resolution = args.resolution
        cage_extrusion = args.cage_extrusion
        max_ray_distance = args.max_ray_distance
        samples = args.samples
        margin = args.margin
        maps_to_bake = args.maps
        img_format = args.format
        do_export = not args.no_export
    else:
        if not CONFIG["highpoly"] or not CONFIG["lowpoly"]:
            print("\n  ERROR: Set highpoly and lowpoly paths in CONFIG or use CLI args.")
            print("  Usage: blender --background --python bake_textures.py -- --highpoly HP.fbx --lowpoly LP.fbx")
            return
        highpoly_path = CONFIG["highpoly"]
        lowpoly_path = CONFIG["lowpoly"]
        output_dir = CONFIG["output"] or os.path.join(os.path.dirname(lowpoly_path), "baked_textures")
        resolution = CONFIG["resolution"]
        cage_extrusion = CONFIG["cage_extrusion"]
        max_ray_distance = CONFIG["max_ray_distance"]
        samples = CONFIG["samples"]
        margin = CONFIG["margin"]
        maps_to_bake = CONFIG["maps"]
        img_format = CONFIG["format"]
        do_export = CONFIG["export_lowpoly"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  High-poly:  {highpoly_path}")
    print(f"  Low-poly:   {lowpoly_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Maps:       {', '.join(maps_to_bake)}")
    print(f"  Samples:    {samples}")
    print(f"  Format:     {img_format}")

    # Step 1: Clear scene
    print("\n[1/7] Clearing scene...")
    clear_scene()

    # Step 2: Setup Cycles
    print("\n[2/7] Setting up Cycles renderer...")
    setup_cycles(samples)

    # Step 3: Import models
    print("\n[3/7] Importing high-poly model...")
    hp_objects = import_model(highpoly_path)
    highpoly = join_objects(hp_objects, name="HighPoly")

    print("\n[4/7] Importing low-poly model...")
    lp_objects = import_model(lowpoly_path)
    lowpoly = join_objects(lp_objects, name="LowPoly")

    # Step 4: Ensure low-poly has UVs
    print("\n[5/7] Checking UVs on low-poly...")
    ensure_uv_unwrap(lowpoly)

    # Step 5: Auto cage extrusion if default
    if cage_extrusion <= 0:
        cage_extrusion = auto_cage_extrusion(highpoly, lowpoly)

    # Step 6: Bake all requested maps
    print("\n[6/7] Baking texture maps...")
    baked_maps = {}
    for map_name in maps_to_bake:
        if map_name not in MAP_CONFIGS:
            print(f"  WARNING: Unknown map type '{map_name}', skipping.")
            continue
        result = bake_map(
            highpoly_obj=highpoly,
            lowpoly_obj=lowpoly,
            map_name=map_name,
            config=MAP_CONFIGS[map_name],
            resolution=resolution,
            cage_extrusion=cage_extrusion,
            max_ray_distance=max_ray_distance,
            margin=margin,
            output_dir=output_dir,
            img_format=img_format,
        )
        baked_maps[map_name] = result

    # Step 7: Assign baked materials and export
    print("\n[7/7] Assigning baked materials...")
    assign_baked_materials(lowpoly, baked_maps, output_dir)

    if do_export:
        export_lowpoly(lowpoly, output_dir, lowpoly_path)

    # Summary
    print("\n" + "=" * 60)
    print("  BAKING COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Maps baked:")
    for map_name, path in baked_maps.items():
        status = "OK" if path else "FAILED"
        print(f"    {map_name:12s} [{status}] {path or ''}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
