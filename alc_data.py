import os
import json
import torch
import opensmile
import numpy as np
import os.path as osp

from time import time
from tqdm import tqdm
from torch.utils.data import Dataset

# TODO Data splitting must be done based on speaker, meaning a specific speaker canont be part of train and test sets
# TODO add OpenSmile caching since it will probably take a lot of time to compute features for each batch

class ALCData(Dataset):

    def __init__(self, data_path = None, transforms = None, max_samples: int = None, verbose: bool = False):
        super().__init__()
        self.ROOT = data_path if data_path else osp.join("data","ALC")
        self.AUDIO_PATH = osp.join(self.ROOT,"wav","h")
        self.LABELS_PATH = osp.join(self.ROOT,"labels","h")
        self.processor = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.class_mapping = {"na": 0, "a": 1}
        self.transforms = transforms
        self.verbose = verbose
        self.max_samples = max_samples
        self.is_data_cached = False

        self.prepare()
    
    def prepare(self):
        """ Prepares the data before training """

        # Read in all files paths
        self.audio_files = sorted([file for file in os.listdir(self.AUDIO_PATH) if file.endswith(".wav")])
        self.label_files = sorted([file for file in os.listdir(self.LABELS_PATH) if file.endswith(".json")])
        assert len(self.audio_files) == len(self.label_files), "Mismatch in number of audio and label files"

        stem_labels = {label_file.removesuffix("_annot.json"): label_file for label_file in self.label_files} 

        # Order independent audio <-> label file mapping
        audio_label_mapping = {}
        for audio_file in self.audio_files:
            audio_stem: str = audio_file.removesuffix(".wav")
            if audio_stem not in stem_labels:
                raise RuntimeError(f"Unmatched audio file: {audio_file}")
            audio_label_mapping[audio_file] = stem_labels[audio_stem]
        
        matched_audio_files = list(audio_label_mapping.keys())
        if self.max_samples:
            matched_audio_files = matched_audio_files[:self.max_samples]

        if self.verbose:
            print(f"Loaded in {len(matched_audio_files)} files ({len(self.audio_files)} total)")
        
        self.files = [] # 0061006001_h_00.wav
        self.class_labels = [] # 0 (NA), 1 (A)
        self.speaker_id_to_index = {} # speaker_id : speaker_index
        for audio_file in tqdm(matched_audio_files):

            label_file = audio_label_mapping[audio_file]

            # Read in label from annot.json
            with open(osp.join(self.LABELS_PATH, label_file), 'r', encoding='utf-8') as file:
                label_config = json.load(file)
                label: str = label_config["levels"][0]["items"][0]["labels"][6]["value"]
                speaker_id = int(audio_file[:3])
                assert label_config["levels"][0]["items"][0]["labels"][6]["name"] == "alc"
                assert label_config["levels"][0]["items"][0]["labels"][2]["name"] == "spn"
                assert speaker_id == int(label_config["levels"][0]["items"][0]["labels"][2]["value"])

                if label == "cna": continue # skip control group class
                
                self.class_labels.append(self.class_mapping[label])
                if speaker_id not in self.speaker_id_to_index:
                    self.speaker_id_to_index[speaker_id] = len(self.speaker_id_to_index)
                self.files.append(audio_file)

        self.class_labels = torch.tensor(self.class_labels, dtype=torch.int64)
        self.len = len(self.class_labels)
    
    def cache(self):
        self.is_data_cached = True
        pass
    
    def calculate_pos_weight(self, train_indices):
        train_labels = self.class_labels[train_indices]
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        if n_pos == 0:
            raise ValueError("Cannot calculate pos_weight with zero positive samples")
        return (n_neg / n_pos).float()
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        audio_file = self.files[index]
        audio_path = osp.join(self.AUDIO_PATH, audio_file)
        speaker_id = int(audio_file[:3])
        class_label = self.class_labels[index]
        speaker_index = self.speaker_id_to_index[speaker_id]
        x = torch.tensor(self.processor.process_file(audio_path).to_numpy(), dtype=torch.float32)
        if self.transforms:
            x = self.transforms(x)
        return x, class_label, speaker_index

    def get_example_sample(self, n: int = 5):
        sample_idx = np.random.default_rng(seed=1999).choice(self.len, size=n, replace=False)
        x_list = []
        y_list = []
        id_list = []
        file_list = []
        for idx in sample_idx:
            x, y, s = self.__getitem__(idx)
            x_list.append(x)
            y_list.append(y)
            id_list.append(s)
            file_list.append(self.files[idx])
        return torch.stack(x_list, dim=0), torch.stack(y_list, dim=0), torch.tensor(id_list), file_list

if __name__ == "__main__":

    t = time()
    data = ALCData(max_samples=500, verbose=True)
    t_tot = time() - t
    print(f"Total time to setup dataset: {t_tot:.2f} s")

    # Get 5 random sample
    x, y, s, files = data.get_example_sample(5)
    print("x-shape:",x.shape," y-shape",y.shape," s-shape",s.shape)
    print("Class labels:",y)
    print("Speaker IDs",s)
    print("Files:",files)


