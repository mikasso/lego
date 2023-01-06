# Flatten all images to be in one directory preserving string category (i.e. "Blocks") label (2357) and name
import os
import shutil

SOURCE = "data/original"
TARGET = "data/mixed"

folders = [f.path for f in os.scandir(SOURCE) if f.is_dir()]
for folder in folders:
    print(f"Copying content of {folder} to {TARGET}..")
    subfolder = [f.path for f in os.scandir(folder) if f.is_dir()]
    for subsubfolder in subfolder:
        files = [f.path for f in os.scandir(subsubfolder) if f.is_file()]
        for file in files:
            file = file.replace("\\", "/")
            parts = file.split("/")
            target_filepath = "_".join(parts[-3:])
            shutil.copy(file, f"{TARGET}/{target_filepath}")
