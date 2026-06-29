
import os
import time
import shutil
import os.path as osp

# Directories
ROOT_DIR = osp.join("data","ALC")
METADATA_DIR = osp.join(ROOT_DIR,"metadata")
TABLE_DIR = osp.join(METADATA_DIR,"CLARINDocu","TABLE")

dry_run = False # False means does actual copying
is_verbose = False
idx = 1

for (root,dirs,files) in os.walk(ROOT_DIR,topdown=True):

    if "metadata" in root:
        continue

    if is_verbose:
        print("Directory path: %s"%root)
        print("Directory Names: %s"%dirs)
        print("Files Names: %s"%files) 
        time.sleep(6)

    for file in files:

        source_path = osp.join(root,file)

        if not file.endswith((".par",".TextGrid",".wav",".json")): continue
        audio_channel_path = "h" if ("_h_" in file) else "m"

        if file.endswith(".wav"): # audio file
            target_path = osp.join(ROOT_DIR,"wav",audio_channel_path,file)
        elif file.endswith(".json"): # label file
            target_path = osp.join(ROOT_DIR,"labels",audio_channel_path,file)
        elif file.endswith((".par","TextGrid")):
            target_path = osp.join(ROOT_DIR,"labels","metadata",audio_channel_path,file)
        else:
            raise RuntimeError(f"Unknown file type encounterd: {file}")
    
        if dry_run:
            print(f"[{idx:5}] | Copied file {source_path} -> {target_path}")
        else:
            print(f"[{idx:5}] | Copied file {source_path} -> {target_path}")
            try:
                shutil.copy(source_path,target_path)
            except:
                print(f"File {target_path} already exists")

        idx += 1