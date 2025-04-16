import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
from os.path import join

import torchvision
from PIL.Image import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

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


if __name__ == "__main__":
    process_rgb_videos(join(RGB_ROOT, "videos"), RGB_ROOT)
    # rename_depth_files(DEPTH_ROOT)
