import os
import json
import torch
import opensmile
import numpy as np
import os.path as osp
from tqdm import tqdm
from time import time
from torch.utils.data import Dataset

# TODO fix "cna" bug. We do we care about this class ?!?!? group it with sober or just remove ? 

class ALCData(Dataset):

    def __init__(self, data_path = None, verbose=False):
        super().__init__()
        self.ROOT = data_path if data_path else osp.join("data","ALC")
        self.AUDIO_PATH = osp.join(self.ROOT,"wav","h")
        self.LABELS_PATH = osp.join(self.ROOT,"labels","h")
        self.processor = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Read in all files paths
        self.audio_files = [file for file in os.listdir(self.AUDIO_PATH) if file.endswith(".wav")]
        self.label_files = [file for file in os.listdir(self.LABELS_PATH) if file.endswith(".json")]

        self.audio_stems = [x.strip(".wav") for x in self.audio_files]
        self.label_stems = [x.strip("_annot.json") for x in self.label_files]

        # Check labels are matched
        if not set(self.audio_stems) == set(self.label_stems):
            assert not set(self.audio_stems).difference(set(self.label_stems)), "Unmatched audio and label sets: A - L != ∅"
            assert not set(self.label_stems).difference(set(self.audio_stems)), "Unmatched audio and label sets: L - A != ∅"
            raise AssertionError("Uncaught assertion error")
        
        self.len = len(self.audio_files)
        self.class_mapping = {"na": 0, "a": 1}

        # Read in labels
        print("Reading in labels")
        self.labels = torch.zeros(self.len, dtype=torch.int64)
        for idx, label_file in enumerate(tqdm(self.label_files)):
            assert self.label_stems[idx] == self.audio_stems[idx]
            with open(osp.join(self.LABELS_PATH,label_file), 'r', encoding='utf-8') as file:
                label_config = json.load(file)
                assert label_config["levels"][0]["items"][0]["labels"][6]["name"] == "alc"
                label = label_config["levels"][0]["items"][0]["labels"][6]["value"]
                self.labels[idx] = self.class_mapping[label]

    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        audio_path = osp.join(self.AUDIO_PATH, self.audio_files[index])
        label = self.labels[index]
        x = self.processor(audio_path)
        return torch.tensor(x), label

    def get_example_sample(self, n: int = 5):
        sample_idx = np.random.choice(self.len, size=n, replace=False)
        for idx in sample_idx:
            x, y = self.__getitem__(idx)
        


if __name__ == "__main__":

    t = time()
    data = ALCData(verbose=True)
    t_tot = time() - t
    print(f"Total time to setup dataset: {t_tot:.2f} s")

    # Get 5 random sample
    #x = data.get_example_sample(5)

