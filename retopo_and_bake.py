"""
Blender Retopo + Bake Pipeline
================================
Single high-poly model in -> auto retopo -> UV unwrap -> bake textures
-> export game-ready low-poly with all maps applied.

Retopo methods:
  - decimate     (default) Fast, mixed tri/quad. Best for hard-surface.
  - quadriflow   Slow, all-quad. Best for characters / animation.
  - voxel_remesh Medium, mostly quad. Best for organic sculpts / terrain.

Supported formats: FBX, OBJ, GLB, GLTF, PLY, STL, DAE, ABC, BLEND

Usage (command line):
    # Minimal (decimate at 10%, all defaults)
    blender --background --python retopo_and_bake.py -- --input model.fbx

    # QuadriFlow with specific face count
    blender --background --python retopo_and_bake.py -- \\
        --input sculpture.obj \\
        --retopo-method quadriflow \\
        --target-faces 5000

    # Voxel remesh for organic shapes
    blender --background --python retopo_and_bake.py -- \\
        --input terrain.glb \\
        --retopo-method voxel_remesh \\
        --target-ratio 0.05

Usage (inside Blender scripting tab):
    Set the variables in the CONFIG section below, then run.
"""

import bpy
import bmesh
import os
import sys
import math
import argparse
from pathlib import Path


# ============================================================
# CONFIG - Edit these if running inside Blender's scripting tab
# ============================================================
CONFIG = {
    "input": "",              # Path to high-poly model (leave empty if using CLI args)
    "output": "",             # Output directory (default: next to input)
    "retopo_method": "decimate",   # "decimate", "quadriflow", or "voxel_remesh"
    "target_ratio": 0.1,     # Fraction of original face count (0.0-1.0)
    "target_faces": 0,        # Absolute face count (overrides target_ratio if > 0)
    "voxel_size": 0.0,        # Voxel size for voxel_remesh (0 = auto)
    "resolution": 4096,       # Texture resolution
    "cage_extrusion": 0.0,    # 0 = auto
    "max_ray_distance": 0.0,  # 0 = auto
    "maps": ["normal", "diffuse", "ao", "roughness", "metallic"],
    "format": "PNG",
    "samples": 128,
    "margin": 16,
    "export_lowpoly": True,
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

# QuadriFlow safety thresholds
QUADRIFLOW_WARN_FACES = 500_000
QUADRIFLOW_ABORT_FACES = 5_000_000


# ============================================================
# Argument Parsing & Validation
# ============================================================

def parse_args():
    """Parse command line arguments when run via blender --python."""
    if "--" not in sys.argv:
        return None

    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser(
        description="Retopo + bake pipeline: high-poly in, game-ready low-poly out"
    )
    parser.add_argument("--input", required=True, help="Path to high-poly model")
    parser.add_argument("--output", default="", help="Output directory (default: next to input)")
    parser.add_argument("--retopo-method", default="decimate",
                        choices=["decimate", "quadriflow", "voxel_remesh"],
                        help="Retopology method (default: decimate)")
    parser.add_argument("--target-ratio", type=float, default=0.1,
                        help="Target face ratio 0.0-1.0 (default: 0.1)")
    parser.add_argument("--target-faces", type=int, default=0,
                        help="Absolute target face count (overrides --target-ratio)")
    parser.add_argument("--voxel-size", type=float, default=0.0,
                        help="Voxel size for voxel_remesh (0 = auto)")
    parser.add_argument("--resolution", type=int, default=4096, help="Texture resolution")
    parser.add_argument("--cage-extrusion", type=float, default=0.0,
                        help="Cage extrusion distance (0 = auto)")
    parser.add_argument("--max-ray-distance", type=float, default=0.0,
                        help="Max ray distance (0 = auto)")
    parser.add_argument("--samples", type=int, default=128, help="Bake samples")
    parser.add_argument("--margin", type=int, default=16, help="UV margin in pixels")
    parser.add_argument("--maps", nargs="+",
                        default=["normal", "diffuse", "ao", "roughness", "metallic"],
                        choices=list(MAP_CONFIGS.keys()), help="Maps to bake")
    parser.add_argument("--format", default="PNG", choices=["PNG", "TIFF", "OPEN_EXR"],
                        help="Output image format")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip exporting low-poly with materials")
    return parser.parse_args(argv)


def validate_inputs(input_path, retopo_method, target_ratio, target_faces, voxel_size):
    """Fail-fast input validation."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = Path(input_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if retopo_method not in ("decimate", "quadriflow", "voxel_remesh"):
        raise ValueError(f"Unknown retopo method: {retopo_method}")

    if not (0.0 < target_ratio <= 1.0):
        raise ValueError(f"target_ratio must be in (0.0, 1.0], got {target_ratio}")

    if target_faces < 0:
        raise ValueError(f"target_faces must be >= 0, got {target_faces}")

    if voxel_size < 0:
        raise ValueError(f"voxel_size must be >= 0, got {voxel_size}")


# ============================================================
# Reused from bake_textures.py (verbatim)
# ============================================================

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
    Removes any previous BakeTarget node first so the correct image is always active.
    Returns the image node so it can be selected for baking."""
    # Ensure the object has a material
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="BakeMaterial")
        mat.use_nodes = True
        obj.data.materials.append(mat)

    mat = obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Remove any previous BakeTarget nodes to avoid name collisions
    for node in list(nodes):
        if node.name.startswith("BakeTarget"):
            nodes.remove(node)

    # Create a fresh image texture node for baking
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


# ============================================================
# New / Modified Functions for Retopo Pipeline
# ============================================================

def duplicate_object(obj, new_name):
    """Deep-copy a mesh object. Makes materials single-user so edits
    to the duplicate's material don't affect the original."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup.name = new_name
    dup.data.name = new_name
    # Make materials single-user so lowpoly and highpoly are independent
    bpy.ops.object.make_single_user(material=True)
    return dup


def ensure_highpoly_material(obj):
    """Ensure the high-poly has a Principled BSDF material for PBR baking.
    Models from AI generators (Trellis, etc.) often have no materials or only
    vertex colors. Without a material, diffuse/roughness/metallic bake black."""
    if obj.data.materials:
        # Already has materials — check if any have a Principled BSDF
        for mat_slot in obj.material_slots:
            if mat_slot.material and mat_slot.material.use_nodes:
                for node in mat_slot.material.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        print(f"  HighPoly already has Principled BSDF material.")
                        return
        print(f"  HighPoly has materials but no Principled BSDF, adding one...")
    else:
        print(f"  HighPoly has no materials, creating default PBR material...")

    mat = bpy.data.materials.new(name="HighPoly_DefaultPBR")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # The default new material already has Principled BSDF + Output
    principled = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled = node
            break

    if not principled:
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Set sensible defaults
    principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    principled.inputs["Roughness"].default_value = 0.5
    principled.inputs["Metallic"].default_value = 0.0

    # If vertex colors exist, connect them to Base Color for diffuse baking
    has_vcol = len(obj.data.color_attributes) > 0
    if has_vcol:
        vcol_name = obj.data.color_attributes[0].name
        vcol_node = nodes.new(type='ShaderNodeVertexColor')
        vcol_node.layer_name = vcol_name
        vcol_node.location = (principled.location.x - 300, principled.location.y)
        links.new(vcol_node.outputs["Color"], principled.inputs["Base Color"])
        print(f"  Connected vertex colors '{vcol_name}' to Base Color.")
    else:
        print(f"  No vertex colors found. Using default gray (0.8) for diffuse.")

    # Assign to all empty material slots, or append if none
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        for i, mat_slot in enumerate(obj.material_slots):
            if mat_slot.material is None:
                obj.data.materials[i] = mat


def estimate_target_faces(obj, target_ratio, target_faces):
    """Compute the target face count from ratio or absolute count.
    Returns the target face count as an integer."""
    original_faces = len(obj.data.polygons)
    print(f"  Original face count: {original_faces:,}")

    if target_faces > 0:
        result = target_faces
    else:
        result = max(1, int(original_faces * target_ratio))

    print(f"  Target face count:   {result:,}  "
          f"({result / original_faces * 100:.1f}% of original)")
    return result


def retopologize(obj, method, target_faces, voxel_size=0.0):
    """Dispatcher: retopologize the object using the chosen method.
    Modifies the object in-place and returns it."""
    print(f"\n  Retopo method: {method}")
    print(f"  Target faces:  {target_faces:,}")

    if method == "decimate":
        _retopo_decimate(obj, target_faces)
    elif method == "quadriflow":
        _retopo_quadriflow(obj, target_faces)
    elif method == "voxel_remesh":
        _retopo_voxel_remesh(obj, target_faces, voxel_size)
    else:
        raise ValueError(f"Unknown retopo method: {method}")

    actual = len(obj.data.polygons)
    print(f"  Retopo complete. Actual face count: {actual:,}")
    return obj


def _retopo_decimate(obj, target_faces):
    """Collapse decimate modifier to reach target face count."""
    original_faces = len(obj.data.polygons)
    if original_faces == 0:
        return

    ratio = target_faces / original_faces
    ratio = max(0.0, min(1.0, ratio))

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = ratio
    mod.use_collapse_triangulate = False

    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"  Decimate applied (ratio={ratio:.4f})")


def _clean_mesh_for_remesh(obj):
    """Clean mesh to make it suitable for QuadriFlow/remesh operations.
    Fixes non-manifold geometry, normals, duplicate vertices."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.fill_holes(sides=4)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.select_all(action='DESELECT')
    # Select and dissolve non-manifold edges
    bpy.ops.mesh.select_non_manifold()
    try:
        bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
    except Exception:
        pass
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  Mesh cleanup done: {len(obj.data.polygons):,} faces")


def _retopo_quadriflow(obj, target_faces):
    """QuadriFlow all-quad remesh. Cleans mesh first, falls back to decimate on failure."""
    original_faces = len(obj.data.polygons)

    if original_faces > QUADRIFLOW_ABORT_FACES:
        raise RuntimeError(
            f"QuadriFlow aborted: {original_faces:,} faces exceeds the "
            f"{QUADRIFLOW_ABORT_FACES:,} safety limit. Use 'decimate' method instead, "
            f"or pre-decimate the mesh first."
        )

    if original_faces > QUADRIFLOW_WARN_FACES:
        print(f"  WARNING: QuadriFlow on {original_faces:,} faces will be slow. "
              f"Consider 'decimate' for faster results.")

    # Clean mesh for manifold requirement
    _clean_mesh_for_remesh(obj)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.quadriflow_remesh(
        target_faces=target_faces,
        use_preserve_sharp=True,
        use_preserve_boundary=True,
        use_mesh_symmetry=False,
    )

    actual = len(obj.data.polygons)
    if actual > target_faces * 2:
        print(f"  QuadriFlow failed to reduce faces ({actual:,} vs target {target_faces:,})")
        print(f"  Falling back to decimate...")
        _retopo_decimate(obj, target_faces)
    else:
        print(f"  QuadriFlow remesh complete (target={target_faces:,})")


def _retopo_voxel_remesh(obj, target_faces, voxel_size=0.0):
    """Voxel remesh modifier with auto voxel size computation."""
    if voxel_size <= 0.0:
        # Auto-compute voxel size from bounding box + target face count.
        # Approximate: surface area ~ 6 * (avg_dim/2)^2 for a box.
        # Each voxel face ~ voxel_size^2, so faces ~ surface_area / voxel_size^2.
        dims = obj.dimensions
        surface_area = 2.0 * (dims.x * dims.y + dims.y * dims.z + dims.z * dims.x)
        if surface_area <= 0:
            surface_area = 1.0
        voxel_size = math.sqrt(surface_area / max(target_faces, 1))
        # Clamp to sane range
        voxel_size = max(voxel_size, 0.001)
        print(f"  Auto voxel size: {voxel_size:.6f} "
              f"(bbox: {dims.x:.2f} x {dims.y:.2f} x {dims.z:.2f})")

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Use the remesh modifier
    mod = obj.modifiers.new(name="VoxelRemesh", type='REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = voxel_size
    mod.use_smooth_shade = True

    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"  Voxel remesh applied (voxel_size={voxel_size:.6f})")


def _save_uv_data(obj):
    """Save current UV layer data to a list so it can be restored later."""
    uv_layer = obj.data.uv_layers.active
    if not uv_layer:
        return None
    return [uv_layer.data[i].uv.copy() for i in range(len(uv_layer.data))]


def _restore_uv_data(obj, uv_data):
    """Restore UV layer data from a saved list."""
    if uv_data is None:
        return
    uv_layer = obj.data.uv_layers.active
    if not uv_layer:
        return
    for i in range(min(len(uv_data), len(uv_layer.data))):
        uv_layer.data[i].uv = uv_data[i]


def _try_uv_method(obj, method_name, method_fn):
    """Run a UV unwrap method, measure coverage, return (coverage, uv_data, name)."""
    method_fn()
    bpy.ops.object.mode_set(mode='OBJECT')
    coverage = _measure_uv_coverage(obj)
    uv_data = _save_uv_data(obj)
    print(f"    {method_name:30s} -> {coverage:.1f}% coverage")
    return coverage, uv_data, method_name


def ensure_uv_unwrap(obj, force=False):
    """Ensure the object has UV maps. Tries multiple unwrap strategies and
    keeps whichever produces the best UV coverage.
    If force=True, strips existing UVs and re-unwraps (needed after retopo).

    Strategies tried:
      - Seam-based at multiple sharpness angles (30-85 deg)
      - Smart UV Project at multiple angle limits
      - Cube projection (good for architectural/hard-surface)
      - Lightmap pack (maximizes coverage for baking)
    """
    if force:
        while obj.data.uv_layers:
            obj.data.uv_layers.remove(obj.data.uv_layers[0])
        print(f"  Cleared existing UVs on '{obj.name}' (force=True)")

    if not obj.data.uv_layers:
        face_count = len(obj.data.polygons)
        print(f"  No UVs found on '{obj.name}' ({face_count:,} faces)")

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Ensure a UV layer exists
        obj.data.uv_layers.new(name="UVMap")

        best_coverage = 0.0
        best_uv_data = None
        best_method = ""

        def _run_candidate(name, fn):
            nonlocal best_coverage, best_uv_data, best_method
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            cov, data, mname = _try_uv_method(obj, name, fn)
            if cov > best_coverage:
                best_coverage = cov
                best_uv_data = data
                best_method = mname

        # --- Strategy 1: Seam-based unwrap at multiple sharpness angles ---
        for angle in [30, 45, 60, 75, 85]:
            def _seam_unwrap(a=angle):
                bpy.ops.mesh.mark_seam(clear=True)
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.mesh.select_mode(type='EDGE')
                bpy.ops.mesh.edges_select_sharp(sharpness=math.radians(a))
                bpy.ops.mesh.mark_seam(clear=False)
                bpy.ops.mesh.select_mode(type='FACE')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
                bpy.ops.uv.average_islands_scale()
                bpy.ops.uv.pack_islands(margin=0.002, rotate=True)
            _run_candidate(f"Seam {angle}deg", _seam_unwrap)

        # --- Strategy 2: Smart UV Project at multiple angle limits ---
        for limit in [1.15, 0.79, 0.52]:
            deg_label = int(math.degrees(limit))
            def _smart(lim=limit):
                bpy.ops.uv.smart_project(angle_limit=lim, island_margin=0.001)
                bpy.ops.uv.average_islands_scale()
                bpy.ops.uv.pack_islands(margin=0.002, rotate=True)
            _run_candidate(f"Smart UV ({deg_label}deg)", _smart)

        # --- Strategy 3: Cube projection (great for architectural/hard-surface) ---
        def _cube_project():
            bpy.ops.uv.cube_project(cube_size=1.0)
            bpy.ops.uv.average_islands_scale()
            bpy.ops.uv.pack_islands(margin=0.002, rotate=True)
        _run_candidate("Cube projection", _cube_project)

        # --- Strategy 4: Lightmap pack (maximizes UV space for baking) ---
        try:
            def _lightmap():
                bpy.ops.uv.lightmap_pack(PREF_CONTEXT='ALL_FACES', PREF_PACK_IN_ONE=True,
                                          PREF_NEW_UVLAYER=False, PREF_MARGIN_DIV=0.1)
            _run_candidate("Lightmap pack", _lightmap)
        except Exception:
            print(f"    Lightmap pack                  -> not available")

        bpy.ops.object.mode_set(mode='OBJECT')

        # Restore the best UV layout
        if best_uv_data is not None:
            _restore_uv_data(obj, best_uv_data)

        print(f"  Best UV method: {best_method} ({best_coverage:.1f}% coverage)")
    else:
        print(f"  UVs found on '{obj.name}': {[uv.name for uv in obj.data.uv_layers]}")


def _measure_uv_coverage(obj):
    """Measure approximate UV space coverage as a percentage of the 0-1 UV tile."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    if not uv_layer:
        bm.free()
        return 0.0
    total_area = 0.0
    for face in bm.faces:
        loops = face.loops
        if len(loops) < 3:
            continue
        uvs = [loop[uv_layer].uv for loop in loops]
        # Shoelace formula for polygon area (fan triangulation from first vertex)
        for i in range(1, len(uvs) - 1):
            a = (uvs[i].x - uvs[0].x) * (uvs[i+1].y - uvs[0].y) \
              - (uvs[i+1].x - uvs[0].x) * (uvs[i].y - uvs[0].y)
            total_area += abs(a) * 0.5
    bm.free()
    return total_area * 100.0


def export_lowpoly(lowpoly_obj, output_dir, original_input_path):
    """Export the low-poly model with baked materials.
    Output named {stem}_lowpoly_baked.{ext}."""
    base_name = Path(original_input_path).stem
    ext = Path(original_input_path).suffix.lower()

    bpy.ops.object.select_all(action='DESELECT')
    lowpoly_obj.select_set(True)
    bpy.context.view_layer.objects.active = lowpoly_obj

    # Export as FBX by default, or match original format
    if ext in (".glb", ".gltf"):
        out_path = os.path.join(output_dir, f"{base_name}_lowpoly_baked.glb")
        bpy.ops.export_scene.gltf(
            filepath=out_path,
            use_selection=True,
            export_format='GLB',
            export_materials='EXPORT',
        )
    elif ext == ".obj":
        out_path = os.path.join(output_dir, f"{base_name}_lowpoly_baked.obj")
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            export_materials=True,
        )
    else:
        out_path = os.path.join(output_dir, f"{base_name}_lowpoly_baked.fbx")
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=True,
            path_mode='COPY',
            embed_textures=True,
        )

    print(f"\n  Exported low-poly with baked textures: {out_path}")
    return out_path


def cleanup_orphan_data():
    """Remove orphan meshes, materials, and images to free memory."""
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    for block in bpy.data.node_groups:
        if block.users == 0:
            bpy.data.node_groups.remove(block)


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  RETOPO + BAKE PIPELINE")
    print("=" * 60)

    # ---- Get configuration from CLI args or CONFIG dict ----
    args = parse_args()
    if args:
        input_path = os.path.abspath(args.input)
        output_dir = args.output or os.path.join(os.path.dirname(input_path), "retopo_baked")
        retopo_method = args.retopo_method
        target_ratio = args.target_ratio
        target_faces = args.target_faces
        voxel_size = args.voxel_size
        resolution = args.resolution
        cage_extrusion = args.cage_extrusion
        max_ray_distance = args.max_ray_distance
        samples = args.samples
        margin = args.margin
        maps_to_bake = args.maps
        img_format = args.format
        do_export = not args.no_export
    else:
        if not CONFIG["input"]:
            print("\n  ERROR: Set 'input' path in CONFIG or use CLI args.")
            print("  Usage: blender --background --python retopo_and_bake.py -- --input model.fbx")
            return
        input_path = os.path.abspath(CONFIG["input"])
        output_dir = CONFIG["output"] or os.path.join(os.path.dirname(input_path), "retopo_baked")
        retopo_method = CONFIG["retopo_method"]
        target_ratio = CONFIG["target_ratio"]
        target_faces = CONFIG["target_faces"]
        voxel_size = CONFIG["voxel_size"]
        resolution = CONFIG["resolution"]
        cage_extrusion = CONFIG["cage_extrusion"]
        max_ray_distance = CONFIG["max_ray_distance"]
        samples = CONFIG["samples"]
        margin = CONFIG["margin"]
        maps_to_bake = CONFIG["maps"]
        img_format = CONFIG["format"]
        do_export = CONFIG["export_lowpoly"]

    # ---- Validate ----
    validate_inputs(input_path, retopo_method, target_ratio, target_faces, voxel_size)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Input:        {input_path}")
    print(f"  Output:       {output_dir}")
    print(f"  Retopo:       {retopo_method}")
    print(f"  Target ratio: {target_ratio}  |  Target faces: {target_faces or 'auto'}")
    print(f"  Resolution:   {resolution}x{resolution}")
    print(f"  Maps:         {', '.join(maps_to_bake)}")
    print(f"  Samples:      {samples}")
    print(f"  Format:       {img_format}")

    try:
        # [1] Clear scene
        print("\n[1/9] Clearing scene...")
        clear_scene()

        # [2] Setup Cycles
        print("\n[2/9] Setting up Cycles renderer...")
        setup_cycles(samples)

        # [3] Import model -> join into "HighPoly"
        print("\n[3/9] Importing model...")
        imported = import_model(input_path)
        highpoly = join_objects(imported, name="HighPoly")
        print(f"  HighPoly: {len(highpoly.data.polygons):,} faces")

        # Ensure highpoly has a PBR material (needed for diffuse/roughness/metallic bake)
        ensure_highpoly_material(highpoly)

        # [4] Duplicate -> "LowPoly"
        print("\n[4/9] Duplicating for retopology...")
        lowpoly = duplicate_object(highpoly, "LowPoly")

        # [5] Retopologize the LowPoly copy
        print("\n[5/9] Retopologizing...")
        target = estimate_target_faces(lowpoly, target_ratio, target_faces)
        retopologize(lowpoly, retopo_method, target, voxel_size)

        # [6] UV unwrap the LowPoly
        print("\n[6/9] UV unwrapping LowPoly...")
        # Decimate preserves original UVs (interpolated). Only force re-unwrap
        # for methods that destroy UVs entirely (quadriflow, voxel_remesh).
        if retopo_method == "decimate" and lowpoly.data.uv_layers:
            preserved_coverage = _measure_uv_coverage(lowpoly)
            print(f"  Decimate preserved original UVs. Coverage: {preserved_coverage:.1f}%")
            if preserved_coverage < 15.0:
                print(f"  Preserved UVs too low, re-unwrapping...")
                ensure_uv_unwrap(lowpoly, force=True)
            else:
                print(f"  Keeping preserved UVs (good coverage).")
        else:
            ensure_uv_unwrap(lowpoly, force=True)

        # [7] Auto-calculate cage extrusion
        print("\n[7/9] Calculating cage extrusion...")
        if cage_extrusion <= 0:
            cage_extrusion = auto_cage_extrusion(highpoly, lowpoly)
        else:
            print(f"  Using user-specified cage extrusion: {cage_extrusion:.4f}")

        # [8] Bake all maps
        print("\n[8/9] Baking texture maps...")
        baked_maps = {}
        for map_name in maps_to_bake:
            if map_name not in MAP_CONFIGS:
                print(f"  WARNING: Unknown map type '{map_name}', skipping.")
                continue
            try:
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
            except Exception as e:
                print(f"  ERROR baking {map_name}: {e}")
                baked_maps[map_name] = None

        # [9] Assign materials + export
        print("\n[9/9] Assigning baked materials & exporting...")
        assign_baked_materials(lowpoly, baked_maps, output_dir)

        if do_export:
            export_lowpoly(lowpoly, output_dir, input_path)

        # Summary
        print("\n" + "=" * 60)
        print("  RETOPO + BAKE COMPLETE")
        print("=" * 60)
        print(f"  Input:     {input_path}")
        print(f"  Output:    {output_dir}")
        print(f"  Method:    {retopo_method}")
        print(f"  HighPoly:  {len(highpoly.data.polygons):,} faces")
        print(f"  LowPoly:   {len(lowpoly.data.polygons):,} faces")
        print(f"  Maps baked:")
        for map_name, path in baked_maps.items():
            status = "OK" if path else "FAILED"
            print(f"    {map_name:12s} [{status}] {path or ''}")
        print("=" * 60 + "\n")

    finally:
        cleanup_orphan_data()


if __name__ == "__main__":
    main()
