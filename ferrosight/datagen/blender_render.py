"""Runs INSIDE Blender. Builds the as-built structure and renders a site walkthrough.

    blender -b -P blender_render.py -- --scene scene.json --out render_dir [options]

Outputs
    frames/frame_0000.jpg ...   RGB walkthrough frames
    gt.npz                      masks  [N,h,w] uint16 (0 = background/clutter, i = id_map[i])
                                depth  [N,h,w] float16, metres along the camera ray
    cameras.json                intrinsics + camera-to-world matrix per frame, and the id map

Masks and depth come from ray casting, so they are exact and independent of render settings.
This file only uses bpy / mathutils / numpy so it can run with a stock Blender install.
"""
import argparse
import json
import math
import os
import random
import sys

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--mask-scale", type=int, default=4, help="mask/depth resolution = image / this")
    p.add_argument("--device", default="AUTO", choices=["AUTO", "GPU", "CPU"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------- materials
def material(name, color, metallic=0.0, roughness=0.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return m


def ground_material(rng):
    m = material("ground", (0.25, 0.2, 0.15), 0.0, 0.95)
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = rng.uniform(0.15, 0.6)
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    base = rng.choice([(0.30, 0.24, 0.17), (0.34, 0.33, 0.31), (0.22, 0.19, 0.15), (0.38, 0.33, 0.24)])
    ramp.color_ramp.elements[0].color = (*[c * 0.6 for c in base], 1)
    ramp.color_ramp.elements[1].color = (*[min(1, c * 1.35) for c in base], 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


# ---------------------------------------------------------------- geometry
def frame_of(start, end, up):
    a, b, u = Vector(start), Vector(end), Vector(up)
    z = b - a
    length = z.length
    z.normalize()
    x = u.cross(z)
    if x.length < 1e-6:
        x = Vector((0, 1, 0)).cross(z)
    x.normalize()
    y = z.cross(x)
    return a, x, y, z, length


def add_member(comp, mat, coll):
    a, x, y, z, length = frame_of(comp["start"], comp["end"], comp["up"])
    outline = comp["outline"]
    n = len(outline)
    verts = [a + x * px + y * py for px, py in outline]
    verts += [v + z * length for v in verts]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
    mesh = bpy.data.meshes.new(comp["id"])
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    obj = bpy.data.objects.new(comp["id"], mesh)
    obj["fs_id"] = comp["id"]
    obj.data.materials.append(mat)
    coll.objects.link(obj)
    return obj


def add_box(name, center, size, mat, coll, rot_z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center, rotation=(0, 0, rot_z))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    obj.data.materials.append(mat)
    return obj


# ---------------------------------------------------------------- scene
def build_scene(data, rng):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    coll = scene.collection
    gx, gy, top = data["grid_x"], data["grid_y"], data["levels"][-1]
    cx, cy = gx[-1] / 2, gy[-1] / 2

    steel_rgb = rng.choice([(0.42, 0.09, 0.06), (0.36, 0.37, 0.39), (0.50, 0.12, 0.07),
                            (0.55, 0.56, 0.58), (0.16, 0.20, 0.26)])
    steel = material("steel", steel_rgb, 0.35, rng.uniform(0.45, 0.75))
    plate = material("plate", (0.30, 0.30, 0.32), 0.5, 0.5)
    for comp in data["components"]:
        add_member(comp, plate if comp["kind"] == "baseplate" else steel, coll)

    t = 0.03  # base plate thickness: slab top sits under the plates
    concrete = material("concrete", (0.52, 0.52, 0.50), 0.0, 0.9)
    add_box("slab", (cx, cy, -t - 0.11), (gx[-1] + 3, gy[-1] + 3, 0.22), concrete, coll)
    bpy.ops.mesh.primitive_plane_add(size=400, location=(cx, cy, -t - 0.2))
    bpy.context.active_object.name = "ground"
    bpy.context.active_object.data.materials.append(ground_material(rng))

    # site clutter: occluders that are NOT on the drawings
    margin = rng.uniform(7.0, 11.0)
    for i in range(rng.randint(3, 8)):
        ang = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(0.55, 0.8)
        px = cx + (gx[-1] / 2 + margin) * rad * math.cos(ang)
        py = cy + (gy[-1] / 2 + margin) * rad * math.sin(ang)
        if -1.5 < px < gx[-1] + 1.5 and -1.5 < py < gy[-1] + 1.5:
            continue
        size = (rng.uniform(1, 3.5), rng.uniform(1, 2.4), rng.uniform(0.6, 2.6))
        col = rng.choice([(0.7, 0.45, 0.1), (0.1, 0.3, 0.55), (0.5, 0.5, 0.5), (0.45, 0.3, 0.15), (0.8, 0.8, 0.75)])
        add_box(f"clutter_{i}", (px, py, -t - 0.2 + size[2] / 2), size, material(f"clutter_{i}", col, 0.1, 0.7),
                coll, rng.uniform(0, math.pi))

    # sun + sky
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = rng.uniform(1.8, 3.2)
    sun.data.angle = math.radians(rng.uniform(0.5, 6.0))
    elev, azim = math.radians(rng.uniform(25, 70)), rng.uniform(0, 2 * math.pi)
    sun.rotation_euler = (math.pi / 2 - elev, 0, azim)
    coll.objects.link(sun)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    scene.world = world
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = rng.uniform(0.25, 0.5)
    bg.inputs["Color"].default_value = (*rng.choice([(0.55, 0.7, 0.95), (0.75, 0.78, 0.82), (0.6, 0.72, 0.9)]), 1)

    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    cam.data.lens = rng.uniform(18, 26)
    cam.data.sensor_width = 36.0
    cam.data.clip_start, cam.data.clip_end = 0.05, 500
    coll.objects.link(cam)
    scene.camera = cam
    return scene, cam, (cx, cy, top, margin)


def camera_path(data, n, rng, info):
    """Perimeter orbit at eye height, then a pass through the building looking up at the framing."""
    cx, cy, top, margin = info
    gx, gy = data["grid_x"], data["grid_y"]
    a, b = gx[-1] / 2 + margin, gy[-1] / 2 + margin
    n_orbit = max(1, int(n * 0.7))
    phase, eye = rng.uniform(0, 2 * math.pi), rng.uniform(1.55, 1.8)
    poses = []
    for i in range(n_orbit):
        t = i / n_orbit
        ang = phase + 2 * math.pi * t
        pos = Vector((cx + a * math.cos(ang), cy + b * math.sin(ang), eye + 0.04 * math.sin(40 * t)))
        tz = top * (0.45 + 0.3 * math.sin(2 * math.pi * 3 * t))
        target = Vector((cx + rng.uniform(-0.3, 0.3), cy + rng.uniform(-0.3, 0.3), tz))
        poses.append((pos, target))
    n_in = n - n_orbit
    y_line = (gy[0] + gy[1]) / 2 if len(gy) > 1 else cy
    for i in range(n_in):
        t = i / max(1, n_in - 1)
        x = -4.0 + (gx[-1] * 0.8 + 4.0) * t        # stop inside so the last frames still see steel
        pos = Vector((x, y_line + 0.3 * math.sin(6 * t), eye))
        yaw = math.radians(55) * math.sin(2 * math.pi * 2 * t)
        pitch = math.radians(12 + 22 * (0.5 + 0.5 * math.sin(2 * math.pi * 3 * t)))
        d = Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
        poses.append((pos, pos + d * 5))
    return poses


def configure_render(scene, args):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    # some Blender builds ship without OpenImageDenoise; rendering must still work there
    scene.cycles.use_denoising = bool(getattr(bpy.app.build_options, "openimagedenoise", False))
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    device = "CPU"
    if args.device in ("AUTO", "GPU"):
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                try:
                    prefs.compute_device_type = kind
                except Exception:
                    continue
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type != "CPU"]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type != "CPU"
                    device = "GPU"
                    print(f"[ferrosight] rendering on {kind}: {[d.name for d in gpus]}")
                    break
        except Exception as exc:
            print(f"[ferrosight] GPU setup failed ({exc}); using CPU")
    scene.cycles.device = device
    if device == "CPU":
        print("[ferrosight] rendering on CPU")


def cast_ground_truth(scene, cam, index_of, w, h):
    """Exact per-pixel component id + distance by ray casting through the camera."""
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    corners = cam.data.view_frame(scene=scene)
    xs, ys = [c.x for c in corners], [c.y for c in corners]
    x0, x1, y0, y1, zc = min(xs), max(xs), min(ys), max(ys), corners[0].z
    rot = cam.matrix_world.to_3x3()
    origin = cam.matrix_world.translation
    mask = np.zeros((h, w), np.uint16)
    depth = np.full((h, w), 500.0, np.float32)
    for v in range(h):
        cy_ = y1 - (v + 0.5) / h * (y1 - y0)
        for u in range(w):
            d = rot @ Vector((x0 + (u + 0.5) / w * (x1 - x0), cy_, zc))
            d.normalize()
            hit, loc, _n, _i, obj, _m = scene.ray_cast(deps, origin, d)
            if hit:
                depth[v, u] = (loc - origin).length
                mask[v, u] = index_of.get(obj.name, 0)
    return mask, depth


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data = json.load(open(args.scene))
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    scene, cam, info = build_scene(data, rng)
    configure_render(scene, args)
    ids = [c["id"] for c in data["components"]]
    index_of = {cid: i + 1 for i, cid in enumerate(ids)}
    mw, mh = args.width // args.mask_scale, args.height // args.mask_scale
    fx = args.width * cam.data.lens / cam.data.sensor_width
    masks, depths, cams = [], [], []
    for i, (pos, target) in enumerate(camera_path(data, args.frames, rng, info)):
        cam.location = pos
        cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        scene.render.filepath = os.path.join(args.out, "frames", f"frame_{i:04d}.jpg")
        bpy.ops.render.render(write_still=True)
        m, d = cast_ground_truth(scene, cam, index_of, mw, mh)
        masks.append(m), depths.append(d.astype(np.float16))
        cams.append(dict(frame=i, cam_to_world=[list(r) for r in cam.matrix_world]))
        print(f"[ferrosight] frame {i + 1}/{args.frames}", flush=True)
    np.savez_compressed(os.path.join(args.out, "gt.npz"), masks=np.stack(masks), depth=np.stack(depths))
    json.dump(dict(width=args.width, height=args.height, mask_scale=args.mask_scale, fx=fx, fy=fx,
                   cx=args.width / 2, cy=args.height / 2, id_map=["<background>"] + ids, frames=cams),
              open(os.path.join(args.out, "cameras.json"), "w"))


if __name__ == "__main__":
    main()
