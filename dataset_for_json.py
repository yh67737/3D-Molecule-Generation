import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData

class JsonFragmentDataset(Dataset):
    """
    A custom PyTorch Dataset to load molecular fragments from individual JSON files.

    Each JSON file is expected to contain the data for a single PyGData object.
    This dataset lazily loads each file upon request (__getitem__).
    """
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): The directory containing all the fragment .json files.
        """
        self.root_dir = root_dir
        # Get a sorted list of all .json file paths to ensure consistent ordering
        self.file_paths = sorted(
            [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.json')]
        )
        if not self.file_paths:
            raise FileNotFoundError(f"No .json files found in the directory: {root_dir}")

    def __len__(self):
        """Returns the total number of fragments (files)."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads and returns a single fragment as a PyGData object.
        
        Args:
            idx (int): The index of the fragment to load.
        
        Returns:
            PyGData objects.
        """
        file_path = self.file_paths[idx]
        
        with open(file_path, 'r') as f:
            data_dict = json.load(f)

        # Reconstruct the PyGData object from the dictionary
        graph_data = PyGData(
            x=torch.tensor(data_dict['x'], dtype=torch.float),
            edge_index=torch.tensor(data_dict['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(data_dict['edge_attr'], dtype=torch.float),
            pos=torch.tensor(data_dict['pos'], dtype=torch.float),
            pring_out = torch.tensor(data_dict['pring_out'], dtype=torch.float),
            is_new_node = torch.tensor(data_dict['is_new_node'], dtype=torch.float)
        )

        return graph_data