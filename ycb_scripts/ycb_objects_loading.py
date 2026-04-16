#!/usr/bin/env python3

# *********
# Authors: Gokul Chenchani
# *********

from omni.isaac.core.utils.stage import add_reference_to_stage
import os

base = "/home/user/chenchani/Desktop/thesis/ws_thesis/src/isaacsim_ycb_dataset_rayst3r/ycb_objects/isaac_objects/1_legacy"

for obj in os.listdir(base):
    usd = os.path.join(base, obj, "isaac_gt.usd")
    if os.path.exists(usd):
        prim_path = f"/World/isaac_objects/{obj}"   # ✅ unique name
        add_reference_to_stage(usd, prim_path)
        
