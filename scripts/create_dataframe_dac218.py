import os
import json
import pandas as pd
import os.path as osp
import soundfile as sf

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, asdict

dry_run = False

# Path variables
ROOT_DIR = osp.join("..","data","SE-DAC218")
METADATA_DIR = osp.join(ROOT_DIR,"metadata")
WAV_DIR = osp.join(ROOT_DIR,"audio","wav")
LABEL_DIR = osp.join(ROOT_DIR,"labels")

df_sessions = pd.read_csv(osp.join(METADATA_DIR,"sessions.csv"))
df_speakers = pd.read_csv(osp.join(METADATA_DIR,"speakers.csv"))

# Feature container dataclass
@dataclass
class DataObject():
    id: str
    wav_file: Path
    speaker_id: str
    session_id: str
    age: int
    sex: str
    class_label: str # na / a
    condition: str # A / B / C / D
    num_sessions: int
    session_utterances: int
    total_utterances: int
    sound_duration: float
    bac_before: float
    bac_after: float
    bac_delta: float


data_dict: dict[str, DataObject] = {}

# Load files
wav_files = sorted(os.listdir(WAV_DIR))

with open(osp.join(LABEL_DIR,"labels.json")) as file:
    labels = json.load(file)
labels = {x["id"]: x["label"] for x in labels}

assert all([(Path(wav_file).stem[:-8] in labels) for wav_file in wav_files]), f""

# Loop though data
for wav_file in tqdm(wav_files): # example file: spk674-sess01a-utt0001

    id = Path(wav_file).stem[:-8]
    label = labels[id]

    session_row = df_sessions.loc[df_sessions["session_id"] == id.replace("-","_"), ["condition", "num_utterances", "total_duration_s"]]
    speaker_row = df_speakers.loc[df_speakers["spk_id"] == label["speaker"], ["gender","age","num_sessions","total_utterances"]]

    # Extract features from config
    speaker_id = label["speaker"]
    session_id = label["session"]
    bac_before = label["bac_before"]
    bac_after = label["bac_after"]
    class_label = label["label"]
    condition = session_row["condition"].values[0]
    num_utterances = session_row["num_utterances"].values[0]
    total_duration_s = session_row["total_duration_s"].values[0]
    sex = speaker_row["gender"].values[0]
    age = speaker_row["age"].values[0]
    num_sessions = speaker_row["num_sessions"].values[0]
    total_utterances = speaker_row["total_utterances"].values[0]
    sound_duration = float(sf.info(osp.join(WAV_DIR,wav_file)).duration)

    # Assertions
    
    # Store data sample instance
    data_dict[id] = DataObject(
        id=id,
        wav_file=wav_file,
        speaker_id=speaker_id,
        session_id=session_id,
        age=age,
        sex=sex,
        class_label=class_label,
        condition=condition,
        num_sessions=num_sessions,
        session_utterances=num_utterances,
        total_utterances=total_utterances,
        sound_duration=sound_duration,
        bac_before=bac_before,
        bac_after=bac_after,
        bac_delta=(bac_before-bac_after)
    )


data_df = pd.DataFrame([asdict(data_obj) for data_obj in data_dict.values()])
print(data_df.head(15))

if not dry_run:
    data_df.to_csv(osp.join(ROOT_DIR,"dac218_samples.csv"), index=False)
