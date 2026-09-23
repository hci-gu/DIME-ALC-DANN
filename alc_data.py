import os
import json
import torch
import random
import hashlib
import opensmile
import numpy as np
import os.path as osp

from time import time
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from hashlib import sha256

def hash_audio_file(file_path):
    with open(file_path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256")
    return digest.hexdigest()


def cache_alc_data(audio_directory_path, cache_path):

    # Read in audio files and sort them
    files = sorted([file for file in os.listdir(audio_directory_path) if file.endswith(".wav")])

    # Define Opensmile pre-processor
    feature_set = opensmile.FeatureSet.ComParE_2016
    feature_level = opensmile.FeatureLevel.Functionals
    processor = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level
    )

    os.makedirs(".cache", exist_ok=True)
    if os.path.exists(cache_path):
        raise RuntimeError(f"Cache file {cache_path} already exists. Delete it to process a new one")
    cache_data_dict = {}
    file_hashes = {}
    for audio_file in tqdm(files):
        audio_path = osp.join(audio_directory_path, audio_file)

        # Pre-process audio file
        x = torch.tensor(processor.process_file(audio_path).to_numpy(), dtype=torch.float32).squeeze(0)
        cache_data_dict[audio_file] = x

        # Calculate hash
        file_hash = hash_audio_file(audio_path)
        file_hashes[audio_file] = file_hash

    # Check that the processed values produce finite values
    if not torch.isfinite(torch.stack(list(cache_data_dict.values()))).all():
        raise RuntimeError("OpenSMILE features contain NaN or infinite values")

    cache_dict = {
        "tensors": cache_data_dict,
        "hashes": file_hashes,
        "feature_set": feature_set.name,
        "feature_level": feature_level.name
    }

    try:
        temporary_path = cache_path + ".tmp"
        torch.save(cache_dict, temporary_path)
        os.replace(temporary_path, cache_path)
        print(f"Successfully saved")
    except Exception as error:
        print(f"Cache save failed: {cache_path}")
        raise RuntimeError("Could not save feature cache") from error


class ALCData(Dataset):

    def __init__(
        self,
        data_path = None,
        transforms = None,
        max_samples: int = None,
        lower_bac_limit: float = None, # promille
        seed: int = 1999,
        verbose: bool = False
        ):
        super().__init__()

        self.ROOT = data_path if data_path else osp.join("data","ALC")
        self.AUDIO_PATH = osp.join(self.ROOT,"wav","h")
        self.LABELS_PATH = osp.join(self.ROOT,"labels","h")
        self.class_mapping = {"na": 0, "a": 1}
        self.transforms = transforms
        self.verbose = verbose
        self.max_samples = max_samples
        self.lower_bac_limit = lower_bac_limit
        self.seed = seed
        self.is_split = False
        self.train_speaker_mapping = {}

        # Prepare dataset
        self.prepare() 
    

    def prepare(self):
        """ Prepares the data before training """

        # Load in cache file
        try:
            cache_path = osp.join(".cache","alc-opensmile-features.pt")
            self.cache_dict = torch.load(cache_path, map_location="cpu")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load in cache file: {e}")

        assert self.cache_dict["feature_set"] == "ComParE_2016"
        assert self.cache_dict["feature_level"] == "Functionals"

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
            generator = random.Random(self.seed)
            selected_files = self.audio_files.copy()
            generator.shuffle(selected_files)
            matched_audio_files = selected_files[:self.max_samples]

        if self.verbose:
            print(f"Loaded in {len(matched_audio_files)} files ({len(self.audio_files)} total)")
        
        self.files = [] # 0061006001_h_00.wav
        self.class_labels = [] # 0 (NA), 1 (A)
        self.speaker_id_to_index = {} # speaker_id : speaker_index (used to map random speakerID to 0,1,2,...,n_speakers-1)
        self.speaker_ids = [] # list of speaker ids (duplicates can occur)
        self.bac_values = [] # blood alcohol concentration in per mille
        for audio_file in tqdm(matched_audio_files):

            # Validate correct hash
            expected_hash = self.cache_dict["hashes"].get(audio_file)
            if expected_hash is None:
                raise FileNotFoundError(f"No cached hash for audio file: {audio_file}")
            file_hash = hash_audio_file(osp.join(self.AUDIO_PATH, audio_file))
            assert expected_hash == file_hash, f"Mismatch in sha256 hash for file: {audio_file}"

            label_file = audio_label_mapping[audio_file]

            # Read in label from annot.json
            with open(osp.join(self.LABELS_PATH, label_file), 'r', encoding='utf-8') as file:
                label_config = json.load(file)
                labels = label_config["levels"][0]["items"][0]["labels"]
                label: str = labels[6]["value"] # "a","na"
                speaker_id = int(audio_file[:3])
                assert labels[6]["name"] == "alc"
                assert labels[2]["name"] == "spn"
                assert speaker_id == int(labels[2]["value"])

                if label == "cna": continue # skip control group class

                labels_by_name = {item["name"]: item["value"] for item in labels}
                try:
                    bac_per_mille = float(labels_by_name["bak"]) * 1000.0
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid or missing BAK value for {audio_file}"
                    ) from error
                if not np.isfinite(bac_per_mille) or bac_per_mille < 0:
                    raise ValueError(
                        f"Invalid BAK value for {audio_file}: {bac_per_mille}"
                    )
                if (self.lower_bac_limit is not None) and (self.lower_bac_limit > bac_per_mille > 0):
                    if self.verbose: print(f"Skipped audio file")
                    continue

                
                self.files.append(audio_file) # list of audio file names
                self.class_labels.append(self.class_mapping[label]) # list of integer class labels
                self.speaker_ids.append(speaker_id) # list of the speaker ids
                self.bac_values.append(bac_per_mille)
                if speaker_id not in self.speaker_id_to_index:
                    self.speaker_id_to_index[speaker_id] = len(self.speaker_id_to_index)

        if self.verbose:
            print(f"Number of data samples used for training & testing: {len(self.class_labels)} (filtered out control group class)")

        self.class_labels = torch.tensor(self.class_labels, dtype=torch.int64)
        self.len = len(self.class_labels)

        # Check for missing files
        missing_files = [audio_file for audio_file in self.files if (audio_file not in self.cache_dict["tensors"])]
        if missing_files:
            raise FileNotFoundError(f"Could not find cached tensor for {len(missing_files)} files.First missing file: {missing_files[0]}")

        # Filter our files
        self.tensors_dict = {audio_file: self.cache_dict["tensors"][audio_file] for audio_file in self.files}

    def calculate_mu_sigma(self, train_indices):
        if len(train_indices) == 0:
            raise ValueError("train_indices must not be empty")

        train_features = torch.stack([self.tensors_dict[self.files[train_idx]] for train_idx in train_indices])
        self.mu = train_features.mean(dim=0)
        self.sigma = train_features.std(dim=0, unbiased=False)
        if self.verbose:
            print(f"Training samples: {len(train_features)} Shape: {train_features.shape}. mu: {self.mu.shape}, sigma: {self.sigma.shape}")
    
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

        expected_classes = set(self.class_mapping.values())
        for split_name, indices in (
            ("train", train_indices),
            ("validation", val_indices),
            ("test", test_indices),
        ):
            present_classes = set(self.class_labels[indices].tolist())
            if present_classes != expected_classes:
                missing_classes = sorted(expected_classes - present_classes)
                raise RuntimeError(f"{split_name} split is missing classes: {missing_classes}")

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

        if not hasattr(self, "mu") or not hasattr(self, "sigma"):
            raise RuntimeError("Call calculate_mu_sigma(train_indices) before accessing samples.")

        # Retrieve relevant helper variables
        audio_file = self.files[index]
        speaker_id = int(audio_file[:3])
        class_label = self.class_labels[index]

        if speaker_id in self.train_speaker_mapping: # This is a train sample
            local_index = self.train_speaker_mapping[speaker_id]
        else: # Val or test sample
            local_index = -1

        metadata = {
            "speaker_id": speaker_id,
            "local_index": local_index,
            "bac": self.bac_values[index]
        }

        try:
            x = self.tensors_dict[audio_file]
        except KeyError as e:
            raise KeyError(f"Could not read in processed tensor for audio file: {audio_file}") from e
        
        # Z-score standardization
        try:
            x = torch.where(self.sigma > 0, (x - self.mu) / self.sigma, torch.zeros_like(x))
        except Exception as e:
            raise RuntimeError(f" Failed to normalize audio tensor. Error: {e}")

        if self.transforms:
            x = self.transforms(x)
        return x, class_label, metadata


    def get_example_sample(self, n: int = 5):
        sample_idx = np.random.default_rng(seed=self.seed).choice(self.len, size=n, replace=False)
        x_list = []
        y_list = []
        id_list = []
        file_list = []
        for idx in sample_idx:
            x, y, metadata = self.__getitem__(idx)
            x_list.append(x)
            y_list.append(y)
            id_list.append(metadata["local_index"])
            file_list.append(self.files[idx])
        return torch.stack(x_list, dim=0), torch.stack(y_list, dim=0), torch.tensor(id_list), file_list



if __name__ == "__main__":

    audio_directory_path = osp.join("data","ALC","wav","h")
    cache_path = osp.join(".cache","alc-opensmile-features.pt")
    cache_alc_data(audio_directory_path, cache_path)

    # print(f"Loading data...")
    # t = time()
    # data = ALCData(
    #     max_samples=None,
    #     verbose=True,
    # )
    # data.cache()
    # t_tot = time() - t
    # print(f"Total time to setup dataset: {t_tot:.2f} s")
    # print(f"Number of data samples: {len(data)}")

    # # Train/Val/Test splitting
    # train_indices, val_indices, test_indices = data.speaker_split(train_frac=0.8, val_frac=0.1, test_frac=0.1)
    # train_data = Subset(data, train_indices)

    # # Get 5 random sample
    # x, y, s, files = train_data.dataset.get_example_sample(5)
    # print("x-shape:",x.shape," y-shape",y.shape," s-shape",s.shape)
    # print("Class labels:",y)
    # print("Local Speaker Index",s)
    # print("Files:",files)
