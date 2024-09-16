# -*- coding: utf-8 -*-

#!pip install vidaug

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

import gdown
import tarfile

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import time

from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.model_selection import train_test_split
import shutil


from keras import callbacks


# !cp "/content/drive/My Drive/New/SSBD.tar.gz" "/content/For_Upload_SSBD.tar.gz"

# import tarfile

# output_file = '/content/For_Upload_SSBD.tar.gz'

# # Extract the contents of the tar.gz file
# with tarfile.open(output_file, 'r:gz') as tar:
#     tar.extractall(path='/content')

# import cv2
# import vidaug.augmentors as va
# import os

# def load_video(path):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     return frames

# def save_frames_to_video(frames, output_video_path, fps=30):
#     height, width, layers = frames[0].shape
#     video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#     for frame in frames:
#         video.write(frame)
#     video.release()

# # Define augmentation sequences
# augmentations = {
#     "random_translate": va.Sequential([va.RandomTranslate()]),
#     "random_rotate": va.Sequential([va.RandomRotate(degrees=10)]),
#     "downsample": va.Sequential([va.Downsample()]),
#     "random_shear": va.Sequential([va.RandomShear(x=0.2, y=0.2)]),
#     "horizontal_flip": va.Sequential([va.HorizontalFlip()]),
#     "vertical_flip": va.Sequential([va.VerticalFlip()]),
#     "invert_color": va.Sequential([va.InvertColor()]),
# }

# # Load video frames
# path = '/content/Spinning_1.mp4'
# frames = load_video(path)

# # Apply each augmentation and save the result
# output_dir = '/content/augmented_videos/'
# os.makedirs(output_dir, exist_ok=True)

# for aug_name, aug_seq in augmentations.items():
#     augmented_frames = aug_seq(frames)
#     output_video_path = os.path.join(output_dir, f"{aug_name}_augmented_video.mp4")
#     save_frames_to_video(augmented_frames, output_video_path, fps=30)
#     print(f"Saved {output_video_path}")

# import cv2
# import vidaug.augmentors as va
# import os
# import numpy as np
# import random

# def custom_random_rotate(frame, angle):
#     h, w = frame.shape[:2]
#     center = (w // 2, h // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
#     return rotated_frame

# def process_and_save_video(video_path, output_path, augmentation, aug_name=""):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

#     # Generate a random angle for rotation if needed
#     if aug_name == "random_rotate":
#         angle = random.uniform(-10, 10)  # Rotation angle between -10 and 10 degrees

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if aug_name == "random_rotate":
#             augmented_frame = custom_random_rotate(frame, angle)
#         elif aug_name == "invert_color":
#             augmented_frame = 255 - frame  # Invert color directly
#         else:
#             augmented_frame = augmentation([frame])[0]  # Apply vidaug augmentations

#         out.write(augmented_frame.astype('uint8'))

#     cap.release()
#     out.release()

# # Rest of the code remains the same


# augmentations = {
#     "horizontal_flip": va.Sequential([va.HorizontalFlip()]),
#     "vertical_flip": va.Sequential([va.VerticalFlip()]),
#     "random_translate": va.Sequential([va.RandomTranslate()]),
#     "random_shear": va.Sequential([va.RandomShear(x=0.2, y=0.2)]),
#     "random_rotate": "random_rotate",  # Custom implementation flag
#     "invert_color": "invert_color",  # Direct operation flag
# }

# video_path = '/content/Spinning_1.mp4'
# output_dir = '/content/augmented_videos/'

# os.makedirs(output_dir, exist_ok=True)

# for aug_name, aug in augmentations.items():
#     output_video_path = os.path.join(output_dir, f"{aug_name}_augmented_video.mp4")
#     print(f"Processing {aug_name} augmentation...")
#     process_and_save_video(video_path, output_video_path, aug, aug_name)
#     print(f"Processed and saved: {output_video_path}")

# import cv2
# import vidaug.augmentors as va
# import os
# import numpy as np
# import random


# def load_video(path):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     return frames

# def custom_random_rotate(frame, angle):
#     h, w = frame.shape[:2]
#     center = (w // 2, h // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
#     return rotated_frame


# def process_and_save_video(video_path, output_path, augmentation, aug_name="", downsample_factor=1):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Adjust FPS for downsample augmentation
#     if aug_name == "downsample":
#         fps = fps / downsample_factor

#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

#     # Generate a random angle for rotation if needed
#     angle = 0
#     if aug_name == "random_rotate":
#         angle = random.uniform(-10, 10)  # Rotation angle between -10 and 10 degrees

#     frame_counter = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Handle custom augmentation cases
#         if aug_name == "random_rotate":
#             augmented_frame = custom_random_rotate(frame, angle)
#         elif aug_name == "invert_color":
#             augmented_frame = 255 - frame
#         elif aug_name == "downsample" and frame_counter % downsample_factor != 0:
#             frame_counter += 1
#             continue
#         elif callable(augmentation):  # Check if the augmentation is callable
#             augmented_frame = augmentation([frame])[0]
#         else:
#             # If the augmentation is not callable and not a handled special case, skip it
#             augmented_frame = frame

#         # Write the augmented frame to the output video
#         out.write(augmented_frame)
#         frame_counter += 1

#     cap.release()
#     out.release()

# # Specify downsample factor here, e.g., 2 for half the frame rate
# downsample_factor = 2

# # Modify your augmentations dictionary to include or exclude 'downsample' as needed
# augmentations = {
#     "horizontal_flip": va.Sequential([va.HorizontalFlip()]),
#     "vertical_flip": va.Sequential([va.VerticalFlip()]),
#     "random_translate": va.Sequential([va.RandomTranslate()]),
#     "random_shear": va.Sequential([va.RandomShear(x=0.2, y=0.2)]),
#     "random_rotate": "random_rotate",
#     "invert_color": "invert_color",
#     "downsample": "downsample",  # Added downsample as an option
# }

# video_path = '/content/Spinning_1.mp4'
# output_dir = '/content/augmented_videos/'

# os.makedirs(output_dir, exist_ok=True)


# for aug_name, aug in augmentations.items():
#     output_video_path = os.path.join(output_dir, f"{aug_name}_augmented_video.mp4")
#     print(f"Processing {aug_name} augmentation...")
#     if aug_name == "downsample":
#         process_and_save_video(video_path, output_video_path, aug, aug_name, downsample_factor)
#     else:
#         process_and_save_video(video_path, output_video_path, aug, aug_name)
#     print(f"Processed and saved: {output_video_path}")

import cv2
import numpy as np
import os
import random
import vidaug.augmentors as va

def custom_random_rotate(frame, angle):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
    return rotated_frame

# def custom_shear(frame, shear_level=0.2):
#     rows, cols = frame.shape[:2]
#     M = np.float32([
#         [1, shear_level, -shear_level * cols / 2],
#         [0, 1, 0]
#     ])
#     sheared_frame = cv2.warpAffine(frame, M, (cols, rows), borderMode=cv2.BORDER_REFLECT101)
#     return sheared_frame


def upsample_frame(frame, scale_factor=1.5):
    height, width = frame.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    upsampled_frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_CUBIC)
    return upsampled_frame

def process_and_save_video(video_path, output_path, augmentation, aug_name="", downsample_factor=1, shear_level=None, upsample_factor=1.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Adjust for "upsample" augmentation
    if aug_name == "upsample":
        width, height = int(width * upsample_factor), int(height * upsample_factor)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
    angle = 25 if aug_name == "random_rotate" else 0

    # Set a random shear level for "random_shear" augmentation
    # if aug_name == "random_shear":
    #     shear_level = random.uniform(-0.2, 0.2)  # Random shear level between -0.2 and 0.2

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if aug_name == "random_rotate":
            frame = custom_random_rotate(frame, angle)
        elif aug_name == "invert_color":
            frame = 255 - frame
        # elif aug_name == "random_shear":
        #     frame = custom_shear(frame, shear_level)
        elif aug_name == "upsample":
            frame = upsample_frame(frame, upsample_factor)
        elif isinstance(augmentation, va.Sequential):
            frame = augmentation([frame])[0]

        if aug_name != "downsample" or frame_counter % downsample_factor == 0:
            out.write(frame)
        frame_counter += 1

    cap.release()
    out.release()

def process_videos_in_folder(input_folder, output_folder, augmentations, downsample_factor=2, upsample_factor=1.5):
    
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    for video_file in video_files:
        base_file_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(input_folder, video_file)

        for aug_name, aug in augmentations.items():
            output_video_name = f"{base_file_name}_{aug_name}.mp4"
            output_video_path = os.path.join(output_folder, output_video_name)
            print(f"Applying {aug_name} augmentation to {video_file}...")
            process_and_save_video(video_path, output_video_path, aug, aug_name, downsample_factor, upsample_factor=upsample_factor)
            print(f"Processed and saved: {output_video_path}")

# Define your augmentations here
augmentations = {
    "horizontal_flip": va.Sequential([va.HorizontalFlip()]),
    "vertical_flip": va.Sequential([va.VerticalFlip()]),
    "upsample": "upsample",
    # "random_shear": "random_shear",  # Custom function, now with consistent random shear per video
    "random_rotate": "random_rotate",
    "invert_color": "invert_color",
    "downsample": "downsample",
}

# For ArmFlapping Data
#input_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD/ArmFlapping'
#output_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD_Augmentation/ArmFlapping'

#process_videos_in_folder(input_folder, output_folder, augmentations)

# For HeadBanging Data
#input_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD/HeadBanging/'
#output_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD_Augmentation/HeadBanging/'

#process_videos_in_folder(input_folder, output_folder, augmentations)

# For Spinning Data
#input_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD/Spinning/'
#output_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD_Augmentation/Spinning/'

#process_videos_in_folder(input_folder, output_folder, augmentations)

input_folder = '/data/home/brijendra/trapti/TypicalDeveloped'
output_folder = '/data/home/brijendra/trapti/TypicalDeveloped_aug'

process_videos_in_folder(input_folder, output_folder, augmentations)

"""Code to copy original data into Augmented folder"""

import os
import shutil

def copy_videos(source_dir, dest_dir, extensions=['.mp4']):
    """
    Copies video files from one directory to another.

    Parameters:
    - source_dir: Path to the source directory.
    - dest_dir: Path to the destination directory.
    - extensions: List of video file extensions to copy. Defaults to ['.mp4'].
    """
    # Check if destination directory exists, if not, create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Initialize a counter to keep track of the number of files copied
    files_copied = 0

    # Loop through all files in the source directory
    for file_name in os.listdir(source_dir):
        # Check if the file matches any of the specified extensions
        if any(file_name.endswith(ext) for ext in extensions):
            # Construct full file path
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(dest_dir, file_name)
            # Copy the video file to the destination directory
            shutil.copy(source_file, destination_file)
            files_copied += 1
            print(f"Copied: {file_name}")

    print(f"Total files copied: {files_copied}")

# Set your directories here
#input_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD/ArmFlapping/'
#output_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD_Augmentation/ArmFlapping/'

# Call the function with your directories
#copy_videos(input_folder, output_folder, ['.mp4', '.avi'])

# Set your directories here
#input_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD/HeadBanging/'
#output_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD_Augmentation/HeadBanging/'

# Call the function with your directories
#copy_videos(input_folder, output_folder, ['.mp4', '.avi'])

# Set your directories here
#input_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD/Spinning/'
#output_folder = '/content/drive/MyDrive/New_SSBD/Yolo5_Augmentation_Data/SSBD_Augmentation/Spinning/'

# Call the function with your directories
#copy_videos(input_folder, output_folder, ['.mp4', '.avi'])