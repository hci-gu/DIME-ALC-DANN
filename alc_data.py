import os
import json
import torch
import opensmile
import numpy as np
import os.path as osp

from time import time
from tqdm import tqdm
from torch.utils.data import Dataset, Subset


class ALCData(Dataset):

    def __init__(
        self,
        data_path = None,
        transforms = None,
        max_samples: int = None,
        seed: int = 1999,
        verbose: bool = False
        ):
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
        self.seed = seed
        self.is_cached = False
        self.is_split = False
        self.train_speaker_mapping = {}

        # Prepare dataset
        self.prepare() 
    

    def prepare(self):
        """ Prepares the data before training """

        # Read in all files from paths
        self.audio_files = sorted([file for file in os.listdir(self.AUDIO_PATH) if file.endswith(".wav")])
        self.label_files = sorted([file for file in os.listdir(self.LABELS_PATH) if file.endswith(".json")])
        assert len(self.audio_files) == len(self.label_files), "Mismatch in number of audio and label files"

        label_stems = {label_file.removesuffix("_annot.json"): label_file for label_file in self.label_files} 

        # Order independent audio <-> label file mapping
        audio_label_mapping = {} # 0061006001_h_00.wav -> 0061006001_h_00_annot.json
        for audio_file in self.audio_files:
            audio_stem: str = audio_file.removesuffix(".wav")
            if audio_stem not in label_stems:
                raise RuntimeError(f"Unmatched audio file: {audio_file}")
            audio_label_mapping[audio_file] = label_stems[audio_stem] 
        
        matched_audio_files = list(audio_label_mapping.keys())
        if self.max_samples:
            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(matched_audio_files), generator=generator).tolist()
            matched_audio_files = [matched_audio_files[i] for i in perm]
            matched_audio_files = matched_audio_files[:self.max_samples]

        if self.verbose:
            print(f"Loaded in {len(matched_audio_files)} files ({len(self.audio_files)} total)")
        
        self.files = [] # 0061006001_h_00.wav
        self.class_labels = [] # 0 (NA), 1 (A)
        self.speaker_id_to_index = {} # speaker_id : speaker_index (used to map random speakerID to 0,1,2,...,n_speakers-1)
        self.speaker_ids = [] # list of speaker ids (duplicates can occur)
        for audio_file in tqdm(matched_audio_files):

            label_file = audio_label_mapping[audio_file]

            # Read in label from annot.json
            with open(osp.join(self.LABELS_PATH, label_file), 'r', encoding='utf-8') as file:
                label_config = json.load(file)
                label: str = label_config["levels"][0]["items"][0]["labels"][6]["value"] # "a","na"
                speaker_id = int(audio_file[:3])
                assert label_config["levels"][0]["items"][0]["labels"][6]["name"] == "alc"
                assert label_config["levels"][0]["items"][0]["labels"][2]["name"] == "spn"
                assert speaker_id == int(label_config["levels"][0]["items"][0]["labels"][2]["value"])

                if label == "cna": continue # skip control group class
                
                self.files.append(audio_file) # list of audio file names
                self.class_labels.append(self.class_mapping[label]) # list of integer class labels
                self.speaker_ids.append(speaker_id) # list of the speaker ids
                if speaker_id not in self.speaker_id_to_index:
                    self.speaker_id_to_index[speaker_id] = len(self.speaker_id_to_index)

        if self.verbose:
            print(f"Number of data samples used for training & testing: {len(self.class_labels)} (filtered out control group class)")

        self.class_labels = torch.tensor(self.class_labels, dtype=torch.int64)
        self.len = len(self.class_labels)
    

    def cache(self, train_indices=None):

        # Look in .cache and see if the data is stored load and return
        os.makedirs(".cache", exist_ok=True)
        cache_path = osp.join(".cache","opensmile-features.pt")
        if osp.exists(cache_path):

            if self.verbose: print(f"Loading pre-processed audio features from cache: {cache_path}")

            # Load from disk
            all_features = torch.load(cache_path, map_location="cpu")

            # Check for missing files
            missing_files = [audio_file for audio_file in self.files if audio_file not in all_features]
            if missing_files:
                raise FileNotFoundError(
                    f"Could not find cached tensor for {len(missing_files)} files. "
                    f"First missing file: {missing_files[0]}"
                )

            # Filter our files
            self.cache_dict = {
                audio_file: all_features[audio_file]
                for audio_file in self.files
            }

            del all_features

        else: # Calculate features

            if self.verbose: print(f"Calculating and caching audio preprocessing")
            self.cache_dict = {}
            feature_tensor = [] # Use to compute Z-score standardization constants
            for audio_file in tqdm(self.files):
                audio_path = osp.join(self.AUDIO_PATH, audio_file)
                x = torch.tensor(self.processor.process_file(audio_path).to_numpy(), dtype=torch.float32).squeeze(0)
                feature_tensor.append(x)
                self.cache_dict[audio_file] = x
            feature_tensor = torch.stack(feature_tensor)


            # If file does not exist save to disk for future
            if self.max_samples is None:
                torch.save(self.cache_dict, cache_path)
            elif self.verbose:
                print("Skipping persistent feature cache because max_samples is set")

        # Calculate mu, sigma based on train indices
        feature_tensor = torch.stack([self.cache_dict[file] for file in self.files])
        train_features = feature_tensor if train_indices is None else feature_tensor[train_indices]
        self.mu = train_features.mean(dim=0)
        self.sigma = train_features.std(dim=0)
    
        if self.verbose: print("Z-score shapes:", self.mu.shape, self.sigma.shape)
        self.is_cached = True

    def calculate_pos_weight(self, train_indices):
        train_labels = self.class_labels[train_indices]
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        if n_pos == 0:
            raise ValueError("Cannot calculate pos_weight with zero positive samples")
        return (n_neg / n_pos).float()
    
    def speaker_split(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ):

        assert abs(train_frac+val_frac+test_frac-1.0) < 1e-6

        unique_speakers = torch.tensor(sorted(set(self.speaker_ids)), dtype=torch.long)
        if self.verbose: print(f"Unique speakers from speaker ID {len(unique_speakers)}")

        # Permutations
        generator = torch.Generator().manual_seed(self.seed)
        n_speakers = len(unique_speakers)
        perm = torch.randperm(n_speakers, generator=generator)

        unique_speakers = unique_speakers[perm]

        n_train = max(1, int(n_speakers * train_frac))
        n_val = max(1, int(n_speakers * val_frac))

        self.train_speakers_id = set(unique_speakers[:n_train].tolist())
        self.val_speakers_id = set(unique_speakers[n_train:(n_train+n_val)].tolist())
        self.test_speakers_id = set(unique_speakers[(n_train+n_val):].tolist())

        # Use for training to map speaker_id -> local_idx \in {0,1,...,len(train_speakers_id)-1} for CE-loss calculation
        self.train_speaker_mapping = {speaker_idx: local_idx for (local_idx,speaker_idx) in enumerate(sorted(self.train_speakers_id))}
        
        train_indices = []
        val_indices = []
        test_indices = []

        for idx, speaker_id in enumerate(self.speaker_ids):
            if speaker_id in self.train_speakers_id:
                train_indices.append(idx)
            elif speaker_id in self.val_speakers_id:
                val_indices.append(idx)
            elif speaker_id in self.test_speakers_id:
                test_indices.append(idx)
            else:
                raise RuntimeError(f"Could not assign speaker ID: {speaker_id} to a split")

        self.is_split = True
        return train_indices, val_indices, test_indices


    def get_split_speakers(self) -> dict[str,list]:
        if not self.is_split: raise RuntimeError("Call speaker_split() before get_split_speakers()")
        return {
            "train_speakers": sorted(self.train_speakers_id),
            "val_speakers": sorted(self.val_speakers_id),
            "test_speakers": sorted(self.test_speakers_id),
        }

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        audio_file = self.files[index]
        speaker_id = int(audio_file[:3])
        class_label = self.class_labels[index]

        if speaker_id in self.train_speaker_mapping: # This is a train sample
            local_index = self.train_speaker_mapping[speaker_id]
        else: # Val or test sample
            local_index = -1

        if self.is_cached:
            x = self.cache_dict[audio_file]
        else:
            audio_path = osp.join(self.AUDIO_PATH, audio_file)
            x = torch.tensor(self.processor.process_file(audio_path).to_numpy(), dtype=torch.float32).squeeze(0)
        
        # Z-score standardization
        x = torch.where(self.sigma > 0, (x - self.mu) / self.sigma, torch.zeros_like(x))

        if self.transforms:
            x = self.transforms(x)
        return x, class_label, local_index


    def get_example_sample(self, n: int = 5):
        sample_idx = np.random.default_rng(seed=self.seed).choice(self.len, size=n, replace=False)
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

    print(f"Loading data...")
    t = time()
    data = ALCData(
        max_samples=None,
        verbose=True,
    )
    data.cache()
    t_tot = time() - t
    print(f"Total time to setup dataset: {t_tot:.2f} s")
    print(f"Number of data samples: {len(data)}")

    # Train/Val/Test splitting
    train_indices, val_indices, test_indices = data.speaker_split(train_frac=0.8, val_frac=0.1, test_frac=0.1)
    train_data = Subset(data, train_indices)

    # Get 5 random sample
    x, y, s, files = train_data.dataset.get_example_sample(5)
    print("x-shape:",x.shape," y-shape",y.shape," s-shape",s.shape)
    print("Class labels:",y)
    print("Local Speaker Index",s)
    print("Files:",files)
