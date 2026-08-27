import os
import re
import json
import time
import shutil
import os.path as osp
import pandas as pd

# Directories
ROOT_DIR = osp.join("..","data","SE-DAC218")
METADATA_DIR = osp.join(ROOT_DIR,"metadata")

# read in csv files
df_sessions = pd.read_csv(osp.join(METADATA_DIR,"sessions.csv"))
df_speakers = pd.read_csv(osp.join(METADATA_DIR,"speakers.csv"))
df_utterances = pd.read_csv(osp.join(METADATA_DIR,"utterances.csv"))

dry_run = False # False means does actual copying
is_verbose = False
idx = 1
delta_limit = 0.1
max_delta = 0.0

label_target_path = osp.join(ROOT_DIR,"labels","labels.json")
label_cache = set()
labels_list = []

for (root,dirs,files) in os.walk(ROOT_DIR,topdown=True):

    if "metadata" in root or "wav" in root:
        continue

    if is_verbose:
        print("Directory path: %s"%root)
        print("Directory Names: %s"%dirs)
        print("Files Names: %s"%files) 
        time.sleep(6)

    for file in files:

        source_path = osp.join(root,file)

        if not file.endswith(".wav"): continue

        if file.endswith(".wav"): # audio file

            print(file, source_path)
            # Parse audio file
            split_path = root.split(os.sep)
            speaker = [x for x in split_path if x.startswith("spk")]
            session = [x for x in split_path if x.startswith("sess")]
            assert len(speaker) == 1, f"Expected only a single match both got {speaker}"
            assert len(session) == 1, f"Expected only a single match both got {session}"
            file_cleaned = file.replace("_bpillar","")
            modified_file = speaker[0]+"-"+session[0]+"-"+file_cleaned
            target_path = osp.join(ROOT_DIR,"audio","wav",modified_file)

            # Label Construct label file
            data_id = f"{speaker[0]}_{session[0]}" # all utterances in this session share same condition & BAC
            if data_id not in label_cache:
                label = {}
                label["speaker"] = speaker[0]
                label["session"] = session[0]
                label_data = df_sessions.loc[df_sessions["session_id"] == data_id, ["condition", "bac_before", "bac_after"]].values[0]
                label["label"] = "na" if (label_data[0] == "A") else "a"
                label["bac_before"] = label_data[1]
                label["bac_after"] = label_data[2]
                if (delta := (label_data[1] - label_data[2])) > delta_limit:
                    max_delta = max(max_delta, delta)
                label_cache.add(data_id)
                labels_list.append({"id": data_id.replace("_","-"), "label": label})

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

print("max delta in data:",max_delta)

# Finally write the labels file
print(f"Saved labels in {label_target_path}")
if not dry_run: 
    with open(label_target_path, "w") as json_file:
        json.dump(labels_list, json_file, indent=4)
