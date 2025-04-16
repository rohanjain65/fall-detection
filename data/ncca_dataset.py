import os
from os.path import join
from typing import Callable, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

ID_TO_LABEL = ["Empty", "Standing", "Sitting", "Lying", "Bending", "Crawling"]
NUM_CLASSES = 6


class NCCADataset(Dataset):
    """
    Dataset class for the Fall Detection dataset using NCCA data.

    Expects the dataset to be structured as follows:
    root/
    ├── split/
        ├── video-1/
            ├── rgb/
                ├── rgb_0001.jpg
                ├── rgb_0002.jpg
                ...
            ├── depth/
                ├── depth_0001.jpg
                ├── depth_0002.jpg
                ...
            ├── labels.csv
        ├── video-2/
            ├── rgb/
            ├── depth/
            ├── labels.csv
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

        self.root = join(root, split)
        self.modality = modality
        self.transformations = transformations

        # Load the dataset metadata
        self.video_paths = [join(self.root, video) for video in os.listdir(self.root)]
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

        for video_index, video in enumerate(video_paths):
            # Load the image labels
            labels = pd.read_csv(join(video, "labels.csv"))["class"]

            # Replace 6 with 0 for the "Empty" class, TODO: Remove once we create a preprocessing script
            labels = labels.replace(6, 0).tolist()

            # Create a (video_index, frame_index, label) tuple for each frame. Frame indices start at 1.
            video_data = [(video_index, frame_index + 1, label) for frame_index, label in enumerate(labels)]

            data.extend(video_data)

        return data

    def calculate_class_frequencies(self) -> Tensor:
        """
        Calculate the frequency of each class in the dataset.

        Returns:
            frequencies (Tensor): A tensor containing the frequency of each class.
        """

        frequencies = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        for _, _, label in self.data:
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
        image_path = join(self.video_paths[video_index], self.modality, f"{self.modality}_{frame_index:04d}.png")

        # Load the image
        image = Image.open(image_path).convert("RGB")

        image = F.to_dtype(F.to_image(image), torch.float32) / 255.0

        # Apply transformations if provided
        if self.transformations:
            image = self.transformations(image)

        return image, class_id
