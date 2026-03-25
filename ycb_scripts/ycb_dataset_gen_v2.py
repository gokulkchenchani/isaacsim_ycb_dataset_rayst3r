import os
import random
import numpy as np
import torch
from PIL import Image

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, Usd

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

USD_PATH = "/home/user/chenchani/Desktop/thesis/isaac_worlds/ycb_world_v1.usd"

OUTPUT_DIR = "/home/user/chenchani/Desktop/thesis/isaac_worlds/ycb_world_v2_dataset"

NUM_FRAMES = 100
MIN_OBJECTS = 2
MAX_OBJECTS = 6

RESOLUTION = (640,480)

CAMERA_PATH = "/World/camera1/Camera1"
YCB_PARENT = "/World/props/YCB"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------
# LOAD STAGE
# ------------------------------------------------

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

# ------------------------------------------------
# FIND YCB OBJECTS
# ------------------------------------------------

ycb_parent = stage.GetPrimAtPath(YCB_PARENT)

objects = []

for child in ycb_parent.GetChildren():
    objects.append(child)

print("Detected objects:", [o.GetName() for o in objects])

# ------------------------------------------------
# CAMERA
# ------------------------------------------------

render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)

rgb = rep.AnnotatorRegistry.get_annotator("rgb")
depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
mask = rep.AnnotatorRegistry.get_annotator("instance_segmentation")

rgb.attach(render_product)
depth.attach(render_product)
mask.attach(render_product)

rep.orchestrator.run()

# ------------------------------------------------
# CAMERA MATRICES
# ------------------------------------------------

def get_camera_data():

    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    xform = UsdGeom.Xformable(cam_prim)

    cam2world = np.array(xform.ComputeLocalToWorldTransform(0))

    intrinsics = np.array([
        [600,0,320],
        [0,600,240],
        [0,0,1]
    ])

    return cam2world, intrinsics

# ------------------------------------------------
# RANDOM OBJECT VISIBILITY
# ------------------------------------------------

def sample_objects():

    k = random.randint(MIN_OBJECTS, MAX_OBJECTS)

    selected = random.sample(objects, k)

    for obj in objects:

        imageable = UsdGeom.Imageable(obj)

        if obj in selected:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

# ------------------------------------------------
# SAVE FRAME
# ------------------------------------------------

def save_frame(i):

    frame_dir = os.path.join(OUTPUT_DIR, f"{i:05d}")
    os.makedirs(frame_dir, exist_ok=True)
    print("Dataset location: ", frame_dir)
    rgb_data = rgb.get_data()
    depth_data = depth.get_data()
    mask_dict = mask.get_data()

    mask_data = mask_dict["data"]

    Image.fromarray(rgb_data).save(os.path.join(frame_dir, "rgb.png"))

    # safe depth normalization
    if np.max(depth_data) > 0:
        depth_norm = (depth_data / np.max(depth_data) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_data).astype(np.uint8)

    Image.fromarray(depth_norm).save(os.path.join(frame_dir, "depth.png"))

    Image.fromarray(mask_data.astype(np.uint8)).save(os.path.join(frame_dir, "mask.png"))

    np.save(os.path.join(frame_dir, "depth.npy"), depth_data)

    cam2world, intrinsics = get_camera_data()

    torch.save(torch.tensor(cam2world), os.path.join(frame_dir, "cam2world.pt"))
    torch.save(torch.tensor(intrinsics), os.path.join(frame_dir, "intrinsics.pt"))

# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------

for i in range(NUM_FRAMES):

    print("Frame", i)

    sample_objects()

    rep.orchestrator.step()

    save_frame(i)

print("Dataset generation finished")

simulation_app.close()