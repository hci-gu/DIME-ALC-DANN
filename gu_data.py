import hashlib
import os
import os.path as osp
import re
from pathlib import Path

import numpy as np
import opensmile
import soundfile as sf
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


GU_FOLDERS = {"filtered-control": "na", "filtered-intoxicated": "a"}
GU_FILENAME_PATTERN = re.compile(
    r"(?P<age>\d+)_(?P<sex>[A-Za-z]+)_(?P<speaker_id>\d+)_(?P<utterance_id>\d+)"
)
GU_CACHE_PATH = Path(".cache/gu-opensmile-features.pt")


def discover_gu_audio(data_path: str | Path) -> dict[str, Path]:
    """Find filtered audio with cohort-qualified, portable cache keys."""
    root = Path(data_path)
    audio_paths = {}
    for folder in GU_FOLDERS:
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"GU directory not found: {directory}")
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        if not files:
            raise ValueError(f"No WAV files found in {directory}")
        for path in files:
            if GU_FILENAME_PATTERN.fullmatch(path.stem) is None:
                raise ValueError(f"Unexpected GU filename: {path.name}")
            audio_paths[path.relative_to(root).as_posix()] = path
    return audio_paths

def hash_audio_file(file_path: str | Path) -> str:
    with open(file_path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256")
    return digest.hexdigest()


def cache_gu_data(
    data_path: str | Path = Path("data/GU"),
    cache_path: str | Path = GU_CACHE_PATH,
) -> None:
    """Cache both filtered cohorts under paths relative to the GU root."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        raise RuntimeError(f"Cache file {cache_path} already exists. Delete it to process a new one")
    audio_paths = discover_gu_audio(data_path)

    # Define Opensmile pre-processor
    feature_set = opensmile.FeatureSet.ComParE_2016
    feature_level = opensmile.FeatureLevel.Functionals
    processor = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_data_dict = {}
    file_hashes = {}
    for audio_file, audio_path in tqdm(audio_paths.items()):

        # Pre-process audio file
        x = torch.tensor(processor.process_file(str(audio_path)).to_numpy(), dtype=torch.float32).squeeze(0)
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
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(cache_dict, temporary_path)
        os.replace(temporary_path, cache_path)
        print(f"Successfully saved")
    except Exception as error:
        print(f"Cache save failed: {cache_path}")
        raise RuntimeError("Could not save feature cache") from error


class GUData(Dataset):
    """Filtered GU cohorts with folder labels and unknown (NaN) BAC values.

    Speaker IDs use twice the source ID plus the class label, keeping IDs
    numeric for DataLoader consumers while separating the two cohorts.
    Evaluation should use normalization statistics from the training dataset.
    """

    def __init__(
        self,
        data_path = None,
        transforms = None,
        seed: int = 1999,
        verbose: bool = False,
        include_audio_length: bool = False,
        cache_path: str | Path = GU_CACHE_PATH,
        ):
        super().__init__()

        self.ROOT = data_path if data_path else osp.join("data","GU")
        self.cache_path = Path(cache_path)
        self.class_mapping = {"na": 0, "a": 1}
        self.transforms = transforms
        self.verbose = verbose
        self.include_audio_length = include_audio_length
        self.seed = seed
        self.is_split = False
        self.train_speaker_mapping = {}

        # Prepare dataset
        self.prepare() 
    

    def prepare(self):
        """ Prepares the data before training """

        # Load in cache file
        try:
            self.cache_dict = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            if self.verbose:
                print(f"Loaded in cache | FS: {self.cache_dict["feature_set"]} | FL: {self.cache_dict["feature_level"]} | Files: {len(self.cache_dict["tensors"])}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load in cache file: {e}")

        if (self.cache_dict["feature_set"], self.cache_dict["feature_level"]) != (
            "ComParE_2016", "Functionals"
        ):
            raise ValueError("Unexpected GU OpenSMILE feature configuration")

        # Read in all audio files from paths
        self.audio_paths = discover_gu_audio(self.ROOT)
        self.audio_files = list(self.audio_paths)

        self.files = [] # filtered-control/21_FEMALE_1003_11.wav
        self.class_labels = [] # 0 (NA), 1 (A)
        self.speaker_id_to_index = {} # speaker_id : speaker_index (used to map random speakerID to 0,1,2,...,n_speakers-1)
        self.speaker_ids = [] # list of speaker ids (duplicates can occur)
        self.bac_values = [] # GU has no measured BAC; use NaN for unknown values
        for audio_file in tqdm(self.audio_files):

            # Validate correct hash
            expected_hash = self.cache_dict["hashes"].get(audio_file)
            if expected_hash is None:
                raise FileNotFoundError(f"No cached hash for audio file: {audio_file}")
            file_hash = hash_audio_file(self.audio_paths[audio_file])
            if expected_hash != file_hash:
                raise ValueError(f"Mismatch in sha256 hash for file: {audio_file}")

            path = Path(audio_file)
            label = GU_FOLDERS[path.parts[0]]
            values = GU_FILENAME_PATTERN.fullmatch(path.stem).groupdict()
            # Numeric source IDs are local to each cohort.
            speaker_id = 2 * int(values["speaker_id"]) + self.class_mapping[label]

            self.files.append(audio_file)
            self.class_labels.append(self.class_mapping[label])
            self.speaker_ids.append(speaker_id)
            self.bac_values.append(float("nan"))
            if speaker_id not in self.speaker_id_to_index:
                self.speaker_id_to_index[speaker_id] = len(self.speaker_id_to_index)

        if self.verbose:
            print(f"Number of data samples used for training & testing: {len(self.class_labels)}")

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

    def set_mu_sigma(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

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

        fractions = (train_frac, val_frac, test_frac)
        if any(not np.isfinite(value) or value <= 0 for value in fractions):
            raise ValueError("GU split fractions must be positive and finite")
        if abs(sum(fractions) - 1.0) >= 1e-6:
            raise ValueError("GU split fractions must sum to one")

        # Each GU speaker has one class, so split speakers within each cohort
        # to keep both classes in every split without speaker leakage.
        generator = torch.Generator().manual_seed(self.seed)
        self.train_speakers_id = set()
        self.val_speakers_id = set()
        self.test_speakers_id = set()
        for class_label in sorted(self.class_mapping.values()):
            speakers = sorted({
                speaker for speaker, label in zip(self.speaker_ids, self.class_labels.tolist())
                if label == class_label
            })
            n_speakers = len(speakers)
            if n_speakers < 3:
                raise ValueError(
                    f"GU needs at least three speakers in class {class_label} "
                    "for train/validation/test splitting"
                )
            permutation = torch.randperm(n_speakers, generator=generator).tolist()
            speakers = [speakers[index] for index in permutation]
            n_train = min(max(1, int(n_speakers * train_frac)), n_speakers - 2)
            n_val = min(max(1, int(n_speakers * val_frac)), n_speakers - n_train - 1)
            self.train_speakers_id.update(speakers[:n_train])
            self.val_speakers_id.update(speakers[n_train:n_train + n_val])
            self.test_speakers_id.update(speakers[n_train + n_val:])

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
        speaker_id = self.speaker_ids[index]
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
        if self.include_audio_length:
            metadata["audio_length"] = float(
                sf.info(self.audio_paths[audio_file]).duration
            )

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

    cache_gu_data()
