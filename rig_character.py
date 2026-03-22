"""
Blender Character Rigging Pipeline
====================================
Import a character GLB -> optional retopo -> auto-rig with humanoid armature
-> apply pose preset -> optionally bake pose into mesh -> export GLB.

Designed for Trellis AI-generated characters that need posing for scene assembly.
Scene scripts (rebuild_match_reference.py) import plain GLBs with no armature
knowledge, so --bake-pose flattens the posed armature into the mesh geometry.

Pose presets:
  - standing    Rest pose (no changes)
  - sitting     Formal seated: upper legs -90deg, lower legs +90deg, upright torso
  - sitting_relaxed  Relaxed seated: slight lean back, legs slightly open

Usage:
    blender --background --python rig_character.py -- \\
        --input character.glb --retopo --target-faces 8000 \\
        --pose sitting --bake-pose --export
"""

import bpy
import bmesh
import os
import sys
import math
import argparse
from pathlib import Path
from mathutils import Vector, Matrix, Euler


# ============================================================
# Supported formats (same as retopo_and_bake.py)
# ============================================================
SUPPORTED_EXTENSIONS = {
    ".fbx", ".obj", ".glb", ".gltf", ".ply", ".stl", ".dae", ".abc", ".blend"
}

# ============================================================
# Pose presets: bone_name -> (x_rot, y_rot, z_rot) in degrees
# Rotations are relative to rest pose, applied in XYZ Euler order.
# ============================================================
POSE_PRESETS = {
    "standing": {},  # Rest pose — no rotations
    "sitting": {
        "upper_leg.L": (-90, 0, 0),
        "upper_leg.R": (-90, 0, 0),
        "lower_leg.L": (90, 0, 0),
        "lower_leg.R": (90, 0, 0),
    },
    "sitting_relaxed": {
        "upper_leg.L": (-85, 0, -8),
        "upper_leg.R": (-85, 0, 8),
        "lower_leg.L": (80, 0, 0),
        "lower_leg.R": (80, 0, 0),
        "spine.001": (-5, 0, 0),  # Slight lean back
    },
}


# ============================================================
# Argument Parsing
# ============================================================

def parse_args():
    """Parse CLI arguments when run via blender --python."""
    if "--" not in sys.argv:
        return None

    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser(
        description="Character rigging pipeline: import -> rig -> pose -> export"
    )
    parser.add_argument("--input", required=True, help="Path to character model (GLB/FBX/OBJ)")
    parser.add_argument("--output", default="", help="Output directory (default: next to input)")
    parser.add_argument("--retopo", action="store_true", help="Apply retopology before rigging")
    parser.add_argument("--retopo-method", default="decimate", choices=["decimate", "quadriflow"],
                        help="Retopology method (default: decimate)")
    parser.add_argument("--target-faces", type=int, default=8000,
                        help="Target face count for retopo (default: 8000)")
    parser.add_argument("--pose", default="standing", choices=list(POSE_PRESETS.keys()),
                        help="Pose preset to apply (default: standing)")
    parser.add_argument("--bake-pose", action="store_true",
                        help="Bake posed armature into mesh geometry (removes armature)")
    parser.add_argument("--export", action="store_true",
                        help="Export final result as GLB")
    return parser.parse_args(argv)


# ============================================================
# Scene Setup
# ============================================================

def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)


def import_model(filepath):
    """Import a 3D model file. Returns list of imported mesh objects."""
    filepath = os.path.abspath(filepath)
    ext = Path(filepath).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

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

    new_objects = [obj for obj in bpy.data.objects
                   if obj.name not in existing and obj.type == 'MESH']

    if not new_objects:
        raise RuntimeError(f"No mesh objects imported from: {filepath}")

    print(f"  Imported {len(new_objects)} mesh object(s): {[o.name for o in new_objects]}")
    return new_objects


def join_objects(objects, name="Character"):
    """Join multiple mesh objects into one."""
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


# ============================================================
# Retopology (simplified from retopo_and_bake.py)
# ============================================================

def retopologize(obj, method, target_faces):
    """Retopologize the mesh to improve auto-weight quality."""
    original_faces = len(obj.data.polygons)
    print(f"  Original faces: {original_faces:,}")
    print(f"  Target faces:   {target_faces:,}")
    print(f"  Method:         {method}")

    if method == "decimate":
        ratio = target_faces / max(original_faces, 1)
        ratio = max(0.0, min(1.0, ratio))

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        mod.use_collapse_triangulate = False
        bpy.ops.object.modifier_apply(modifier=mod.name)

    elif method == "quadriflow":
        # Clean mesh first for manifold requirement
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.fill_holes(sides=4)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        try:
            bpy.ops.object.quadriflow_remesh(
                target_faces=target_faces,
                use_preserve_sharp=True,
                use_preserve_boundary=True,
                use_mesh_symmetry=False,
            )
        except RuntimeError:
            print("  QuadriFlow failed, falling back to decimate...")
            retopologize(obj, "decimate", target_faces)
            return

    actual = len(obj.data.polygons)
    print(f"  Retopo complete: {actual:,} faces")


# ============================================================
# Armature Creation
# ============================================================

def create_humanoid_armature(mesh_obj):
    """Create a 15-bone humanoid armature sized to fit the mesh bounding box.

    Bones: root, hips, spine, spine.001, neck, head,
           shoulder.L/R, upper_arm.L/R, lower_arm.L/R,
           upper_leg.L/R, lower_leg.L/R, foot.L/R
    """
    # Get mesh bounding box in world space
    bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    min_co = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_co = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    center = (min_co + max_co) / 2.0
    dims = max_co - min_co

    height = dims.z
    width = dims.x
    depth = dims.y

    # Proportional bone positions (fractions of total height from bottom)
    foot_h = height * 0.02
    ankle_h = height * 0.05
    knee_h = height * 0.25
    hip_h = height * 0.45
    spine_h = height * 0.55
    chest_h = height * 0.65
    shoulder_h = height * 0.75
    neck_h = height * 0.80
    head_top_h = height * 0.98

    # X offsets for limbs
    hip_x = width * 0.12
    shoulder_x = width * 0.22
    elbow_x = width * 0.35
    hand_x = width * 0.48

    # Y offset (front) for slight forward offset of arms
    arm_y = depth * 0.02

    base_z = min_co.z
    cx = center.x
    cy = center.y

    print(f"  Mesh bounds: {dims.x:.3f} x {dims.y:.3f} x {dims.z:.3f}")
    print(f"  Creating 15-bone humanoid armature...")

    # Create armature
    arm_data = bpy.data.armatures.new("CharacterArmature")
    arm_obj = bpy.data.objects.new("CharacterArmature", arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = arm_data.edit_bones

    def add_bone(name, head, tail, parent_name=None, connect=False):
        bone = edit_bones.new(name)
        bone.head = Vector(head)
        bone.tail = Vector(tail)
        if parent_name:
            bone.parent = edit_bones[parent_name]
            bone.use_connect = connect
        return bone

    # Spine chain: root -> hips -> spine -> spine.001 -> neck -> head
    add_bone("root", (cx, cy, base_z + foot_h), (cx, cy, base_z + hip_h))
    add_bone("hips", (cx, cy, base_z + hip_h), (cx, cy, base_z + spine_h), "root", connect=True)
    add_bone("spine", (cx, cy, base_z + spine_h), (cx, cy, base_z + chest_h), "hips", connect=True)
    add_bone("spine.001", (cx, cy, base_z + chest_h), (cx, cy, base_z + shoulder_h), "spine", connect=True)
    add_bone("neck", (cx, cy, base_z + neck_h), (cx, cy, base_z + neck_h + height * 0.05), "spine.001")
    add_bone("head", (cx, cy, base_z + neck_h + height * 0.05), (cx, cy, base_z + head_top_h), "neck", connect=True)

    # Left arm: shoulder.L -> upper_arm.L -> lower_arm.L
    add_bone("shoulder.L", (cx, cy, base_z + shoulder_h),
             (cx + shoulder_x, cy, base_z + shoulder_h), "spine.001")
    add_bone("upper_arm.L", (cx + shoulder_x, cy - arm_y, base_z + shoulder_h),
             (cx + elbow_x, cy - arm_y, base_z + shoulder_h - height * 0.02), "shoulder.L")
    add_bone("lower_arm.L", (cx + elbow_x, cy - arm_y, base_z + shoulder_h - height * 0.02),
             (cx + hand_x, cy - arm_y, base_z + shoulder_h - height * 0.04), "upper_arm.L", connect=True)

    # Right arm: shoulder.R -> upper_arm.R -> lower_arm.R
    add_bone("shoulder.R", (cx, cy, base_z + shoulder_h),
             (cx - shoulder_x, cy, base_z + shoulder_h), "spine.001")
    add_bone("upper_arm.R", (cx - shoulder_x, cy - arm_y, base_z + shoulder_h),
             (cx - elbow_x, cy - arm_y, base_z + shoulder_h - height * 0.02), "shoulder.R")
    add_bone("lower_arm.R", (cx - elbow_x, cy - arm_y, base_z + shoulder_h - height * 0.02),
             (cx - hand_x, cy - arm_y, base_z + shoulder_h - height * 0.04), "upper_arm.R", connect=True)

    # Left leg: upper_leg.L -> lower_leg.L -> foot.L
    add_bone("upper_leg.L", (cx + hip_x, cy, base_z + hip_h),
             (cx + hip_x, cy, base_z + knee_h), "root")
    add_bone("lower_leg.L", (cx + hip_x, cy, base_z + knee_h),
             (cx + hip_x, cy, base_z + ankle_h), "upper_leg.L", connect=True)
    add_bone("foot.L", (cx + hip_x, cy, base_z + ankle_h),
             (cx + hip_x, cy - depth * 0.3, base_z + foot_h), "lower_leg.L", connect=True)

    # Right leg: upper_leg.R -> lower_leg.R -> foot.R
    add_bone("upper_leg.R", (cx - hip_x, cy, base_z + hip_h),
             (cx - hip_x, cy, base_z + knee_h), "root")
    add_bone("lower_leg.R", (cx - hip_x, cy, base_z + knee_h),
             (cx - hip_x, cy, base_z + ankle_h), "upper_leg.R", connect=True)
    add_bone("foot.R", (cx - hip_x, cy, base_z + ankle_h),
             (cx - hip_x, cy - depth * 0.3, base_z + foot_h), "lower_leg.R", connect=True)

    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"  Armature created: {len(arm_data.bones)} bones")
    return arm_obj


# ============================================================
# Auto-Weight with Fallbacks
# ============================================================

def parent_mesh_to_armature(mesh_obj, arm_obj):
    """Parent mesh to armature with automatic weights.
    Falls back to envelope weights, then bare parent if auto-weights fail."""
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    # Try automatic weights (bone heat)
    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        print("  Auto-weights (bone heat) applied successfully")
        return
    except RuntimeError as e:
        print(f"  Auto-weights failed: {e}")

    # Fallback 1: Envelope weights
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    try:
        bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
        print("  Fallback: envelope weights applied")
        return
    except RuntimeError as e:
        print(f"  Envelope weights also failed: {e}")

    # Fallback 2: Bare parent (no weights — mesh moves rigidly with root bone)
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.parent_set(type='ARMATURE')
    print("  Fallback: bare armature parent (no weights)")


# ============================================================
# Pose Application
# ============================================================

def apply_pose(arm_obj, pose_name):
    """Apply a pose preset to the armature."""
    preset = POSE_PRESETS.get(pose_name)
    if preset is None:
        raise ValueError(f"Unknown pose preset: {pose_name}. Available: {list(POSE_PRESETS.keys())}")

    if not preset:
        print(f"  Pose '{pose_name}': rest pose (no changes)")
        return

    print(f"  Applying pose: {pose_name}")

    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    for bone_name, (rx, ry, rz) in preset.items():
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            print(f"    WARNING: Bone '{bone_name}' not found, skipping")
            continue

        pbone.rotation_mode = 'XYZ'
        pbone.rotation_euler = Euler((math.radians(rx), math.radians(ry), math.radians(rz)), 'XYZ')
        print(f"    {bone_name}: ({rx}, {ry}, {rz}) deg")

    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  Pose '{pose_name}' applied")


# ============================================================
# Bake Pose into Mesh
# ============================================================

def bake_pose_to_mesh(mesh_obj, arm_obj):
    """Bake the current pose into mesh geometry and remove the armature.
    This produces a static mesh in the posed position — ready for scene assembly
    scripts that just import and place GLBs without armature support."""
    print("  Baking pose into mesh geometry...")

    # Apply armature modifier on the mesh to bake deformation
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    # Find the armature modifier
    arm_mod = None
    for mod in mesh_obj.modifiers:
        if mod.type == 'ARMATURE':
            arm_mod = mod
            break

    if arm_mod:
        # Apply modifier to bake the deformation
        bpy.ops.object.modifier_apply(modifier=arm_mod.name)
        print("  Armature modifier applied (deformation baked)")
    else:
        print("  WARNING: No armature modifier found on mesh")

    # Clear parent (keep transform)
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # Delete the armature
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.delete(use_global=False)

    # Clean up vertex groups left by armature
    mesh_obj.vertex_groups.clear()

    print("  Pose baked. Armature removed. Mesh is now a static posed model.")


# ============================================================
# Export
# ============================================================

def export_glb(mesh_obj, output_path):
    """Export the mesh as GLB."""
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        use_selection=True,
        export_format='GLB',
        export_materials='EXPORT',
    )
    print(f"  Exported: {output_path}")


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  CHARACTER RIGGING PIPELINE")
    print("=" * 60)

    args = parse_args()
    if not args:
        print("\n  ERROR: Use CLI args.")
        print("  Usage: blender --background --python rig_character.py -- --input character.glb --pose sitting --bake-pose --export")
        return

    input_path = os.path.abspath(args.input)
    output_dir = args.output or os.path.dirname(input_path)
    do_retopo = args.retopo
    retopo_method = args.retopo_method
    target_faces = args.target_faces
    pose_name = args.pose
    do_bake_pose = args.bake_pose
    do_export = args.export

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Input:        {input_path}")
    print(f"  Output:       {output_dir}")
    print(f"  Retopo:       {'yes (' + retopo_method + ', ' + str(target_faces) + ' faces)' if do_retopo else 'no'}")
    print(f"  Pose:         {pose_name}")
    print(f"  Bake pose:    {'yes' if do_bake_pose else 'no'}")
    print(f"  Export:       {'yes' if do_export else 'no'}")

    # [1] Clear scene
    print("\n[1/7] Clearing scene...")
    clear_scene()

    # [2] Import model -> join into single mesh
    print("\n[2/7] Importing model...")
    imported = import_model(input_path)
    mesh_obj = join_objects(imported, name="Character")
    print(f"  Mesh: {len(mesh_obj.data.polygons):,} faces")

    # [3] Optional retopo
    if do_retopo:
        print("\n[3/7] Retopologizing...")
        retopologize(mesh_obj, retopo_method, target_faces)
    else:
        print("\n[3/7] Retopo skipped")

    # [4] Create humanoid armature
    print("\n[4/7] Creating humanoid armature...")
    arm_obj = create_humanoid_armature(mesh_obj)

    # [5] Auto-weight mesh to armature
    print("\n[5/7] Parenting mesh to armature (auto-weights)...")
    parent_mesh_to_armature(mesh_obj, arm_obj)

    # [6] Apply pose
    print("\n[6/7] Applying pose...")
    apply_pose(arm_obj, pose_name)

    # [7] Bake pose + export
    print("\n[7/7] Finalize...")

    stem = Path(input_path).stem

    if do_bake_pose:
        bake_pose_to_mesh(mesh_obj, arm_obj)
        suffix = f"_{pose_name}" if pose_name != "standing" else "_posed"
    else:
        suffix = f"_{pose_name}_rigged" if pose_name != "standing" else "_rigged"

    if do_export:
        out_name = f"{stem}{suffix}.glb"
        out_path = os.path.join(output_dir, out_name)
        export_glb(mesh_obj, out_path)

    # Summary
    print("\n" + "=" * 60)
    print("  RIGGING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Input:      {input_path}")
    print(f"  Faces:      {len(mesh_obj.data.polygons):,}")
    print(f"  Pose:       {pose_name}")
    print(f"  Baked:      {'yes' if do_bake_pose else 'no'}")
    if do_export:
        print(f"  Output:     {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
