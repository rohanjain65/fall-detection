import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
from os.path import join

import torchvision
from PIL.Image import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from data.ntu_dataset import parse_action_id

DATA_ROOT = "/data/fall-detection/ntu/processed/"

DEPTH_ROOT = join(DATA_ROOT, "depth")
RGB_ROOT = join(DATA_ROOT, "rgb")
IR_ROOT = join(DATA_ROOT, "ir")


def process_video_file(file_path: str, output_root: str, extension: str = ".avi") -> None:
    """
    Process a single video file and save frames as images.

    Args:
        file_path (str): Path to the video file.
        output_root (str): Directory to save the extracted frames.
        extension (str): Video file extension, default is ".avi".
    """

    # Read the video file
    frames, _, _ = torchvision.io.read_video(file_path, pts_unit="sec", output_format="TCHW")

    # Get the base file name without the extension
    file_name = file_path.split("/")[-1].strip(extension).strip("_rgb")

    # Create the output directory to store the frames
    os.makedirs(join(output_root, file_name), exist_ok=True)

    # Save the frames
    for i, frame in enumerate(frames):
        image: Image = to_pil_image(frame)

        image.save(join(output_root, file_name, f"{i}.png"))


def process_rgb_videos(video_root: str, output_root: str, extension: str = ".avi", num_workers: int = 16) -> None:
    """
    Process videos from the specified directory and save frames as images.

    Args:
        video_root (str): Directory containing the video files.
        output_root (str): Directory to save the extracted frames.
        extension (str): Video file extension to look for, default is ".avi".
    """

    # Create the output directory
    os.makedirs(output_root, exist_ok=True)

    # Find all video files with the specified extension
    file_paths = glob(join(video_root, f"*{extension}"))

    # Process the video files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        func = partial(process_video_file, output_root=output_root, extension=extension)

        list(tqdm(executor.map(func, file_paths), total=len(file_paths), desc=f"Processing {video_root}", unit="file"))


def rename_depth_files(root: str) -> None:
    """
    Rename depth files in the specified directory.

    Converts files from MDepth-00000001.png, MDepth-00000002.png, etc. to 0.png, 1.png, etc.

    Args:
        root (str): Directory containing the depth files.
    """

    video_dirs = [join(root, video_dir) for video_dir in os.listdir(root) if os.path.isdir(join(root, video_dir))]

    for video_dir in tqdm(video_dirs, desc="Renaming depth files", unit="video"):
        image_paths = glob(join(video_dir, "*.png"))

        for image_path in image_paths:
            # Get the file name without the extension
            file_name = os.path.basename(image_path)

            # Get the index from the file name
            index = int(file_name.split("-")[-1].split(".")[0]) - 1

            # Create the new file name
            new_file_name = f"{index}.png"

            # Create the new file path
            new_file_path = os.path.join(video_dir, new_file_name)

            # Rename the file
            os.rename(image_path, new_file_path)


def balance_classes(root: str, *, ratio: int = 4) -> None:
    """
    Balance the number of falling and non-falling videos in the dataset by randomly removing non-falling videos.

    Args:
        root (str): The root directory of the dataset.
        ratio (int): The desired ratio of non-falling to falling videos. Default is 4.

    """
    rgb_root = join(root, "rgb")
    depth_root = join(root, "depth")

    # Get the list of all video names in the RGB directory
    video_names = os.listdir(rgb_root)

    # Separate the video paths into "falling" and "non-falling" categories
    falling_videos = []
    non_falling_videos = []

    for video_name in video_names:
        action_id = parse_action_id(video_name)

        if action_id == 1:
            falling_videos.append(video_name)
        else:
            non_falling_videos.append(video_name)

    # Randomly select 4x the number of falling videos from non-falling videos
    num_videos_to_remove = len(non_falling_videos) - (ratio * len(falling_videos))

    videos_to_remove = random.sample(non_falling_videos, num_videos_to_remove)

    for video_name in tqdm(videos_to_remove, desc="Removing videos", unit="video"):
        rgb_path = join(rgb_root, video_name)
        depth_path = join(depth_root, video_name)

        assert os.path.exists(rgb_path), f"RGB path does not exist: {rgb_path}"
        assert os.path.exists(depth_path), f"Depth path does not exist: {depth_path}"

        shutil.rmtree(rgb_path)
        shutil.rmtree(depth_path)


def train_val_split(root: str, *, train_size: float = 0.75) -> None:
    """
    Splits the dataset into train and val sets.

    Args:
        root (str): The root directory of the dataset.
        train_size (float): The proportion of the dataset to include in the train split.
    """
    rgb_root = join(root, "rgb")
    depth_root = join(root, "depth")

    # Get the list of all video names in the RGB directory
    video_names = os.listdir(rgb_root)

    # Seperate the videos into train and val sets
    train_videos, val_videos = train_test_split(video_names, train_size=train_size)

    # Create the train and val directories if they don't exist
    os.makedirs(join(root, "train", "rgb"), exist_ok=True)
    os.makedirs(join(root, "train", "depth"), exist_ok=True)
    os.makedirs(join(root, "val", "rgb"), exist_ok=True)
    os.makedirs(join(root, "val", "depth"), exist_ok=True)

    # Move the train videos
    for video in tqdm(train_videos, desc="Moving train videos"):
        shutil.move(join(rgb_root, video), join(root, "train", "rgb", video))
        shutil.move(join(depth_root, video), join(root, "train", "depth", video))

    # Move the val videos
    for video in tqdm(val_videos, desc="Moving val videos"):
        shutil.move(join(rgb_root, video), join(root, "val", "rgb", video))
        shutil.move(join(depth_root, video), join(root, "val", "depth", video))

    # Remove the original directories
    shutil.rmtree(rgb_root)
    shutil.rmtree(depth_root)


# if __name__ == "__main__":
# process_rgb_videos(join(RGB_ROOT, "videos"), RGB_ROOT)
# rename_depth_files(DEPTH_ROOT)
