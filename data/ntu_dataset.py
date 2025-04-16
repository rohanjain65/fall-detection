import os
import re
from glob import glob
from os.path import join
from typing import Callable, List, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

ID_TO_LABEL = ["Not Falling", "Falling"]
NUM_CLASSES = 2

FALLING_ACTION_ID = 43


class NTUDataset(Dataset):
    """
    Dataset class for the Fall Detection dataset using NTU data.

    Expects the dataset to be structured as follows:
    root/
    ├── split/
        ├── rgb/
            ├── video-1/
                ├── 0.png
                ├── 1.png
                ...
            ├── video-2/
                ├── 0.png
                ├── 1.png
                ...
        ├── depth/
            ├── video-1/
                ├── 0.png
                ├── 1.png
                ...
            ├── video-2/
                ├── 0.png
                ├── 1.png
    ...

    Args:
        root (str): The path to the root directory of the dataset.
        split (str): The split to use, must be one of 'train', 'val', or 'test'.
        modality (str): The modality to use, must be one of 'rgb' or 'depth'.
        transformations (Callable[[Tensor], Tensor]): Transformations to apply to each image, optional.
    """

    def __init__(
        self,
        root: str,
        split: str,
        modality: str = "rgb",
        transformations: Callable[[Tensor], Tensor] = None,
    ) -> None:

        # Validate the dataset file structure
        assert os.path.exists(root), f"Dataset root {root} does not exist."
        assert split in ["train", "val", "test"], f"Invalid split {split}, must be one of 'train', 'val', or 'test'."
        assert os.path.exists(join(root, split)), f"Split {split} does not exist in dataset root {root}."

        # Validate the modality
        assert modality in ["rgb", "depth"], f"Invalid modality {modality}, must be one of 'rgb' or 'depth'."

        self.modality = modality
        self.root = join(root, split, modality)
        self.transformations = transformations

        # Load the dataset metadata
        self.video_paths = [join(self.root, video) for video in os.listdir(self.root)]
        # self.video_paths = self.video_paths[:1000]  # Limit to 100 videos for testing
        self.data = self._process_data(self.video_paths)

    def _process_data(self, video_paths: List[str]) -> List[Tuple[int, int, int]]:
        """
        Process the dataset to extract image paths and labels.

        Args:
            video_paths (List[str]): List of paths to the videos in the dataset.

        Returns:
            data (List[Tuple[int, int, int]]): List of tuples containing the dataset index, frame index, and class id.
        """

        data = []

        for video_index, video_path in tqdm(enumerate(video_paths), total=len(video_paths), desc="Processing Videos", unit="video"):
            # Count the number of frames in the video directory
            num_frames = len(glob(join(video_path, "*.png")))

            # Parse the action ID from the video path
            label = parse_action_id(video_path)

            # Create a (video_index, frame_index, label) tuple for each frame.
            video_data = [(video_index, frame_index, label) for frame_index in range(num_frames)]

            data.extend(video_data)

        return data

    def calculate_class_frequencies(self) -> Tensor:
        """
        Calculate the frequency of each class in the dataset.

        Returns:
            frequencies (Tensor): A tensor containing the frequency of each class.
        """

        frequencies = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        for _, _, label in tqdm(self.data, desc="Calculating Class Frequencies", unit="frame"):
            frequencies[label] += 1

        return frequencies

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Get the image and class label for the given index.

        Args:
            idx (int): The index of the frame in the concatenated dataset.

        Returns:
            image (Tensor): The image, with shape (3, H, W) if RGB, or (1, H, W) if depth.
            class_id (int): The class id of the frame.
        """

        # Get the video index, frame index, and class id from the data
        video_index, frame_index, class_id = self.data[idx]

        # Construct the image path
        image_path = join(self.video_paths[video_index], f"{frame_index}.png")

        # Load the image
        image = Image.open(image_path).convert("RGB")

        image = F.to_dtype(F.to_image(image), torch.float32) / 255.0

        # Apply transformations if provided
        if self.transformations:
            image = self.transformations(image)

        return image, class_id


def parse_action_id(file_path: str) -> int:
    """
    Parse action ID from the file path. File paths are expected to be in the format: SsssCcccPpppRrrrAaaa

    sss - Setup number
    ccc - Camera ID
    ppp - Subject ID
    rrr - Replication number (1 or 2)
    aaa - Action class label.

    Args:
        video_path (str): Path to the video file.

    Returns:
        action_id (int): Action ID extracted from the video path.
    """

    # Extract the action ID
    match = re.search(r"A(\d{3})", file_path)

    # Check if the match was successful
    assert match is not None, f"Failed to parse action ID from {file_path}"

    action_id = int(match.group(1))

    # Convert to fall/no-fall
    action_id = int(action_id == FALLING_ACTION_ID)

    return action_id
