import torch
from tqdm import tqdm
from torch.utils.data import Dataset



class ALCData(Dataset):

    def __init__(self, data_path):
        super().__init__()

    
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


if __name__ == "__main__":

    data = ALCData()