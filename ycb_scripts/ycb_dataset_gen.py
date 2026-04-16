#!/usr/bin/env python3

# *********
# Authors: Gokul Chenchani
# *********

import os
import random
import numpy as np
import torch
from PIL import Image

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdGeom
from omni.isaac.core import World
import omni.replicator.core as rep

# -------------------------------------------------
# PATH TO YOUR USD SCENE
# -------------------------------------------------

USD_PATH = "/home/user/chenchani/Desktop/thesis/isaac_worlds/ycb_world_v1.usd"

# -------------------------------------------------

OUTPUT_DIR = "/home/user/chenchani/Desktop/thesis/isaac_worlds/ycb_world_v1_dataset"
NUM_FRAMES = 100
RESOLUTION = (640,480)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD SCENE
# -------------------------------------------------

omni.usd.get_context().open_stage(USD_PATH)

stage = omni.usd.get_context().get_stage()

world = World(stage_units_in_meters=1.0)

# -------------------------------------------------
# FIND YCB OBJECTS
# -------------------------------------------------

YCB_PARENT = "/World/props/YCB"

ycb_parent = stage.GetPrimAtPath(YCB_PARENT)

objects = []

for child in ycb_parent.GetChildren():
    objects.append(child)

print("Objects detected:", [o.GetName() for o in objects])

# -------------------------------------------------
# CAMERA
# -------------------------------------------------

CAMERA_PATH = "/World/camera1/Camera1"

render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)

rgb = rep.AnnotatorRegistry.get_annotator("rgb")
depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
mask = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

rgb.attach(render_product)
depth.attach(render_product)
mask.attach(render_product)

rep.orchestrator.run()
# -------------------------------------------------
# TABLE AREA
# -------------------------------------------------

TABLE_CENTER = [0,0,0.75]
TABLE_SIZE = [0.4,0.4]

# -------------------------------------------------
# RANDOMIZE OBJECTS
# -------------------------------------------------

def randomize_objects():

    for prim in objects:

        x = random.uniform(-TABLE_SIZE[0]/2, TABLE_SIZE[0]/2)
        y = random.uniform(-TABLE_SIZE[1]/2, TABLE_SIZE[1]/2)

        pos = (
            TABLE_CENTER[0]+x,
            TABLE_CENTER[1]+y,
            TABLE_CENTER[2]+0.05
        )

        rot = random.uniform(0,360)

        xform = UsdGeom.Xformable(prim)

        xform.ClearXformOpOrder()

        t = xform.AddTranslateOp()
        r = xform.AddRotateZOp()

        t.Set(pos)
        r.Set(rot)

# -------------------------------------------------
# CAMERA MATRICES
# -------------------------------------------------

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

# -------------------------------------------------
# SAVE
# -------------------------------------------------

def save_frame(i):

    frame_dir = os.path.join(OUTPUT_DIR, f"{i:05d}")
    os.makedirs(frame_dir, exist_ok=True)

    rgb_data = rgb.get_data()
    depth_data = depth.get_data()
    mask_data = mask.get_data()

    Image.fromarray(rgb_data).save(os.path.join(frame_dir, "rgb.png"))

    depth_norm = (depth_data / depth_data.max() * 255).astype(np.uint8)
    Image.fromarray(depth_norm).save(os.path.join(frame_dir, "depth.png"))

    Image.fromarray(mask_data.astype(np.uint8)).save(os.path.join(frame_dir, "mask.png"))

    cam2world, intrinsics = get_camera_data()

    torch.save(torch.tensor(cam2world), os.path.join(frame_dir, "cam2world.pt"))
    torch.save(torch.tensor(intrinsics), os.path.join(frame_dir, "intrinsics.pt"))

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------

for i in range(NUM_FRAMES):

    print("Frame",i)

    randomize_objects()

    for _ in range(60):
        world.step(render=True)

    rep.orchestrator.step()

    save_frame(i)

print("DONE")

simulation_app.close()