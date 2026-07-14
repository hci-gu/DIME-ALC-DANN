import os
import json
import pandas as pd
import os.path as osp
import soundfile as sf

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, asdict

# Path variables
ROOT_DIR = osp.join("..","data","ALC")
WAV_H_DIR = osp.join(ROOT_DIR,"wav","h")
LABEL_H_DIR = osp.join(ROOT_DIR,"labels","h")


# Feature container dataclass
@dataclass
class DataObject():
    id: str
    wav_file: Path
    label_file: Path
    speaker_id: str
    session_id: str
    prompt_id: str
    block_id: str
    version: int 
    class_label: str
    age: int
    sex: str
    alcohol_habit: str
    br_ac: float
    bl_ac: float
    sound_duration: float
    car_id: str
    is_control_group: bool
    channel: str = "h"


data_dict: dict[str, DataObject] = {}

# Load files
wav_files = sorted(os.listdir(WAV_H_DIR))
label_files = sorted(os.listdir(LABEL_H_DIR))

# Sanity check
wave_filenames = set([Path(x).stem for x in wav_files])
label_filenames = set([Path(x).stem[:-6] for x in label_files])
assert wave_filenames == label_filenames, "The sets of filenames "

label_files_by_audio_stem = {
    Path(label_file).stem[:-6]: label_file
    for label_file in label_files
}

# Loop though data
for audio_file in tqdm(wav_files): # example file: 0061006001_h_00.wav

    label_file = label_files_by_audio_stem[Path(audio_file).stem]
    underscore_idx = audio_file.index("_")
    id = audio_file[:underscore_idx]

    # Load label config
    with open(osp.join(LABEL_H_DIR,label_file), 'r', encoding='utf-8') as file:
        label_config = json.load(file)
        label = label_config["levels"][0]["items"][0]["labels"]
    
    # Extract features from config
    speaker_id = id[:3]
    session_id = id[3:7]
    block_id = session_id[0]
    prompt_id = id[7:10]
    version = int(audio_file[underscore_idx+3:underscore_idx+5])
    class_label = label[6]["value"]
    sex = label[7]["value"]
    age = label[8]["value"]
    alcohol_habit = label[10]["value"]
    br_ac = label[11]["value"]
    bl_ac = label[12]["value"]
    is_control_group = block_id in ["5","6"]
    car_id = "CAR_B" if block_id in {"1","2","5"} else "CAR_A" # CAR_A = Passat | CAR_B = Open
    sound_duration = float(sf.info(osp.join(WAV_H_DIR,audio_file)).duration)

    # Cross validate with label config
    assert label_config["annotates"] == audio_file
    assert label_config["name"] == Path(audio_file).stem
    assert label[0]["value"] == label_config["name"]
    assert label[1]["value"] == id
    assert label[2]["name"] == "spn"
    assert label[4]["name"] == "item"
    assert label[6]["name"] == "alc"
    assert label[7]["name"] == "sex"
    assert label[8]["name"] == "age"
    assert label[10]["name"] == "drh"
    assert label[11]["name"] == "aak"
    assert label[12]["name"] == "bak"

    # Validate dataset's README structure
    if block_id in {"1","3"}:
        assert label[6]["value"] == "a"
    elif block_id in {"2","4"}:
        assert label[6]["value"] == "na", f"error {block_id} | {label[6]["value"]}"
    elif block_id in {"5","6"}:
        assert label[6]["value"] == "cna", f"error {block_id} | {label[6]["value"]}"
    else:
        raise RuntimeError(f"Unknown block ID: {block_id}")
    
    if class_label == "a":
        assert block_id in {"1","3"}
    elif class_label == "na":
        assert block_id in {"2","4"}
    elif class_label == "cna":
        assert block_id in {"5","6"}
    else:
        raise RuntimeError(f"Unknown block ID: {block_id}")


    # Store data sample instance
    data_dict[id] = DataObject(
        id=id,
        wav_file=audio_file,
        label_file=label_file,
        speaker_id=speaker_id,
        session_id=session_id,
        prompt_id=prompt_id,
        block_id=block_id,
        version=version,
        class_label=class_label,
        sex=sex,
        age=age,
        alcohol_habit=alcohol_habit,
        br_ac=br_ac,
        bl_ac=bl_ac,
        is_control_group=is_control_group,
        car_id=car_id,
        sound_duration=sound_duration
    )


data_df = pd.DataFrame([asdict(data_obj) for data_obj in data_dict.values()])
data_df.to_csv(osp.join(ROOT_DIR,"alc_data_samples.csv"), index=False)
print(data_df.head())
