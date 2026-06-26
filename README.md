# Blender Pipeline

Headless **Blender Python** scripts for turning raw, AI-generated 3D meshes (e.g. Trellis output)
into clean, posable, textured game/film assets. Run them with `blender --background --python <script>`.

## Scripts

### `retopo_and_bake.py` — retopology + texture baking
A 9-step pipeline that decimates / retopologises a dense mesh, **UV-unwraps it**, and bakes textures
onto the new topology. Tries multiple unwrap strategies (seam-based at several angles, Smart UV, cube
projection, lightmap pack) and keeps the best — `pack_islands(rotate=True)` lifts UV coverage from
**~10–20% to ~66%** on architectural meshes. Includes 3 retopo methods with fallbacks for non-manifold
game assets.

> ⚠️ **Known Blender gotcha (handled here):** never re-set `colorspace_settings.name` on a
> generated-type image *after* `bpy.ops.object.bake()` — Blender silently regenerates the image
> from `generated_color` and wipes the baked pixels. Set colourspace once at image creation.

### `rig_character.py` — auto-rigging
Import a character GLB → optional retopo → auto-rig with a humanoid armature → apply a pose preset
(`standing` / `sitting` / `sitting_relaxed`) → optionally **bake the pose into the mesh** → export GLB.
Built for AI-generated characters that need posing before scene assembly.

### `bake_textures.py` — standalone texture bake
The original bake routine, kept for reference.

## Usage

```bash
blender --background --python retopo_and_bake.py -- --input model.glb --output out.glb
blender --background --python rig_character.py    -- --input char.glb  --pose sitting --bake-pose
```

## Notes

- Tested against dense Trellis AI meshes (1M+ faces, no materials).
- QuadriFlow retopo needs manifold input — there's a decimate fallback for game assets that aren't.
