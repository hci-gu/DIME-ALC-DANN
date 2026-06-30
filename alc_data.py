import os
import json
import torch
import opensmile
import numpy as np
import os.path as osp

from time import time
from tqdm.contrib import tzip
from torch.utils.data import Dataset

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
        self.index_mapping = {}
        self.class_mapping = {"na": 0, "a": 1}
        self.transforms = transforms

        # Read in all files paths
        self.audio_files = [file for file in os.listdir(self.AUDIO_PATH) if file.endswith(".wav")]
        self.label_files = [file for file in os.listdir(self.LABELS_PATH) if file.endswith(".json")]
        assert len(self.audio_files) == len(self.label_files), "Mismatch in number of audio and label files"

        if verbose:
            print(f"Loaded in {len(self.audio_files)} audio files and {len(self.label_files)} label files")

        # Limit the number of samples we read int
        if max_samples:
            self.audio_files[:max_samples]
            self.label_files[:max_samples]

        self.audio_stems = []
        self.label_stems = []
        self.files = []
        self.files_labels = []
        self.labels = []
        for (audio_file, label_file) in tzip(self.audio_files, self.label_files):

            # Stemming
            audio_stem = audio_file.strip(".wav")
            label_stem = label_file.strip("_annot.json")
            assert audio_stem == label_stem, f"Stem missmatch: Audio: {audio_stem} | Label: {label_stem}"

            self.audio_stems.append(audio_stem)
            self.label_stems.append(label_stem)

            # Read in label from annot.json
            with open(osp.join(self.LABELS_PATH, label_file), 'r', encoding='utf-8') as file:
                label_config = json.load(file)
                assert label_config["levels"][0]["items"][0]["labels"][6]["name"] == "alc"
                label: str = label_config["levels"][0]["items"][0]["labels"][6]["value"]

                if label == "cna": continue
                
                self.labels.append(self.class_mapping[label])
                self.files_labels.append(label_file)
                self.files.append(audio_file)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)
        self.len = len(self.labels)

        # Check labels are matched
        if not set(self.audio_stems) == set(self.label_stems):
            assert not set(self.audio_stems).difference(set(self.label_stems)), "Unmatched audio and label sets: A - L != ∅"
            assert not set(self.label_stems).difference(set(self.audio_stems)), "Unmatched audio and label sets: L - A != ∅"
            raise AssertionError("Uncaught assertion error")
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        audio_path = osp.join(self.AUDIO_PATH, self.files[index])
        label = self.labels[index]
        x = self.processor.process_file(audio_path).to_numpy()
        return torch.tensor(x), label

    def get_example_sample(self, n: int = 5):
        sample_idx = np.random.default_rng(seed=1999).choice(self.len, size=n, replace=False)
        x_list = []
        y_list = []
        file_list = []
        for idx in sample_idx:
            x, y = self.__getitem__(idx)
            x_list.append(x)
            y_list.append(y)
            file_list.append((self.files[idx], self.files_labels[idx]))
        return torch.stack(x_list, dim=0), torch.stack(y_list, dim=0), file_list

if __name__ == "__main__":


    t = time()
    data = ALCData(max_samples=500, verbose=True)
    t_tot = time() - t
    print(f"Total time to setup dataset: {t_tot:.2f} s")

    # Get 5 random sample
    x, y, files = data.get_example_sample(5)
    print(x.shape)
    print(y)
    print(y.shape)
    print(files)


