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
import omni.replicator.core as rep
from pxr import UsdGeom, Usd

# ---------------- CONFIG ----------------

# PROVIDE PATH FOR USD FILE AS PER YOUR SYSTEM

USD_PATH = "src/isaacsim_ycb_dataset_rayst3r/isaac_scenes/ycb_new_objects_world_v1.usd"
OUTPUT_DIR = "./../ycb_gen_dataset"

NUM_FRAMES = 30
MIN_OBJECTS = 12
MAX_OBJECTS = 25

RESOLUTION = (640,480)

CAMERA_PATH = "/World/camera1/Camera1"
YCB_PARENT = "/World/props/isaac_objects"

DEPTH_MAX_METERS = 10.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD SCENE ----------------

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

# ---------------- FIND OBJECTS ----------------

ycb_parent = stage.GetPrimAtPath(YCB_PARENT)

objects = []
for child in ycb_parent.GetChildren():
    objects.append(child)

print("Detected objects:", [o.GetName() for o in objects])

# ---------------- ASSIGN SEMANTICS ----------------

from omni.isaac.core.utils.semantics import add_update_semantics

for obj in objects:
    add_update_semantics(obj, obj.GetName())

# ---------------- CAMERA ----------------

render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)

rgb = rep.AnnotatorRegistry.get_annotator("rgb")
depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")

sem = rep.AnnotatorRegistry.get_annotator(
    "semantic_segmentation",
    init_params={"colorize": False}
)

rgb.attach(render_product)
depth.attach(render_product)
sem.attach(render_product)

rep.orchestrator.run()

# ---------------- CAMERA MATRICES ----------------

def get_camera_data():

    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    cam = UsdGeom.Camera(cam_prim)

    focal = float(cam.GetFocalLengthAttr().Get())
    h_ap = float(cam.GetHorizontalApertureAttr().Get())
    v_ap = float(cam.GetVerticalApertureAttr().Get())

    width,height = RESOLUTION

    fx = (focal/h_ap)*width
    fy = (focal/v_ap)*height
    cx = width*0.5
    cy = height*0.5

    intrinsics = np.array([
        [fx,0,cx],
        [0,fy,cy],
        [0,0,1]
    ],dtype=np.float32)

    imageable = UsdGeom.Imageable(cam_prim)

    cam2world = np.array(
        imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float32
    )

    return cam2world,intrinsics

# ---------------- OBJECT SAMPLING ----------------

def sample_objects():

    k = random.randint(MIN_OBJECTS,MAX_OBJECTS)
    selected = random.sample(objects,k)

    for obj in objects:

        imageable = UsdGeom.Imageable(obj)

        if obj in selected:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

# ---------------- DEPTH NORMALIZATION ----------------

def depth_to_png(depth_raw):

    depth = np.asarray(depth_raw)

    if depth.ndim == 3:
        depth = depth[:,:,0]

    depth = np.nan_to_num(depth)

    depth = np.clip(depth,0,DEPTH_MAX_METERS)

    depth_mm = (depth*1000).astype(np.uint16)

    return depth_mm

# ---------------- MASK BUILDER ----------------

def build_mask(sem_raw):

    if isinstance(sem_raw,dict):

        sem_data = sem_raw["data"]
        info = sem_raw["info"]

    else:

        sem_data = sem_raw
        info = {}

    sem_data = np.asarray(sem_data)

    if sem_data.ndim == 3:
        sem_data = sem_data[:,:,0]

    mask = (sem_data>0).astype(np.uint8)*255

    return mask

# ---------------- SAVE FRAME ----------------

def save_frame(i):

    frame_dir = os.path.join(OUTPUT_DIR,f"{i:05d}")
    os.makedirs(frame_dir,exist_ok=True)

    rgb_data = rgb.get_data()
    depth_raw = depth.get_data()
    sem_raw = sem.get_data()

    rgb_img = np.asarray(rgb_data)[...,:3].astype(np.uint8)

    depth_png = depth_to_png(depth_raw)

    mask = build_mask(sem_raw)

    Image.fromarray(rgb_img).save(os.path.join(frame_dir,"rgb.png"))
    Image.fromarray(depth_png).save(os.path.join(frame_dir,"depth.png"))
    Image.fromarray(mask).save(os.path.join(frame_dir,"mask.png"))

    cam2world,intrinsics = get_camera_data()

    torch.save(torch.tensor(cam2world),os.path.join(frame_dir,"cam2world.pt"))
    torch.save(torch.tensor(intrinsics),os.path.join(frame_dir,"intrinsics.pt"))

# ---------------- MAIN LOOP ----------------

for i in range(NUM_FRAMES):

    print("Frame",i)

    sample_objects()

    rep.orchestrator.step()
    rep.orchestrator.step()

    save_frame(i)

print("Dataset generation finished")

simulation_app.close()