import os
import random
import shutil
from pathlib import Path

def split_png_files(source_folder, num_splits=5, train_ratio=0.8):
    # Resolve source folder and parent folder
    source_folder = Path(source_folder)
    parent_folder = source_folder.parent
    splits_folder = parent_folder / "splits"

    # Create splits folder
    splits_folder.mkdir(exist_ok=True)

    # Get all PNG files in the source folder
    png_files = list(source_folder.glob("*.png"))
    if not png_files:
        print("No PNG files found in the source folder.")
        return

    # Shuffle files randomly
    random.shuffle(png_files)

    # Create split folders
    for split_num in range(1, num_splits + 1):
        split_folder = splits_folder / f"split_{split_num}"
        train_folder = split_folder / "train"
        val_folder = split_folder / "val"

        # Create train and val folders
        train_folder.mkdir(parents=True, exist_ok=True)
        val_folder.mkdir(parents=True, exist_ok=True)

        #create Ir and Vis folders
        ir_train, vis_train = train_folder / "Vis", train_folder / "Ir"
        ir_val, vis_val = val_folder / "Vis", val_folder / "Ir"
        ir_train.mkdir(parents=True, exist_ok=True)
        ir_val.mkdir(parents=True, exist_ok=True)
        vis_train.mkdir(parents=True, exist_ok=True)
        vis_val.mkdir(parents=True, exist_ok=True)

        # Split files into train and val
        split_point = int(len(png_files) * train_ratio)
        train_files = png_files[:split_point]
        val_files = png_files[split_point:]

        # Copy files to respective folders
        for file in train_files:
            shutil.copy(parent_folder / "Vis" / file.name, vis_train / file.name)
            shutil.copy(parent_folder / "Ir" / file.name, ir_train / file.name)
        for file in val_files:
            shutil.copy(parent_folder / "Vis" / file.name, vis_val / file.name)
            shutil.copy(parent_folder / "Ir" / file.name, ir_val / file.name)

        # Rotate files for the next split
        png_files = png_files[-split_point:] + png_files[:-split_point]

    print(f"Splitting complete. Splits are saved in '{splits_folder}'.")

# Usage example
source_folder = "/mnt/store/jparanj1/Thermal_Datasets/M3FD_Fusion/Vis"
split_png_files(source_folder)
