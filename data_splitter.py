'''
 # @ Author: Kai Ruben Enerhaugen
 # @ Email: kai.r.h.enerhaugen@nmbu.no
 # @ Create Time: 2025-11-14
 # @ Modified time: 2025-11-18
 '''

#TODO Restructure folders

from pathlib import Path
import re
import shutil
import random

# Settings
SOURCE_DIR = Path('./DJI_202508171710_007_Create-Area-Route31/')    #TODO Define a default filestructure instead of this.
OUTPUT_DIR = Path('./output')
DATA_DIR = Path("./dataset")
IMAGE_FILETYPE = '.jpg'
DESTINATION_DIRS = ['train', 'val', 'test']
TRAINING_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2

# Make sure ratios sum to 1.0
assert abs(TRAINING_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1!"

# renaming labels so they match the correct image
def rename_label(path: Path):
    m = re.search(r"(DJI_.*)", path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return m.group(1)


# Get all images and labels
images = [i for i in SOURCE_DIR.iterdir() if i.is_file() and i.suffix.lower() in {IMAGE_FILETYPE}]
labels = [i for i in Path("YOLOv8_TrainingData/labels/").iterdir() if i.is_file() and i.suffix.lower() == '.txt']

# Shuffle
random.shuffle(images)

# Check if output directory is empty
if any((OUTPUT_DIR).iterdir()):
    raise FileExistsError("Output directory is not empty!")

for folder in ["images", "labels"]:
    (DATA_DIR / folder / "train").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / folder / "val").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / folder / "test").mkdir(parents=True, exist_ok=True)

for folder, subfolder in zip(["images", "labels"], DESTINATION_DIRS):
    (DATA_DIR / folder / subfolder).mkdir(parents=True, exist_ok=True)  



#####################################################
#TODO Match labels after cleaning the filename
#####################################################
# pairs = []
# for image, label in zip(images, labels):
#     if image.stem == clean_names(label.stem):
#         pairs.append((image, label))

##################################################


# find corresponding labels 
pairs = []
for label in labels:
    m = re.search(r"(\d+_D)$", label.stem)
    for image in images:
        if image.stem.endswith(m.group(1)):
            pairs.append((image, label))


# train, eval, test split
ntrain = int(len(images) * TRAINING_RATIO)
nvalidation = int(ntrain + len(images) * VALIDATION_RATIO)
training = pairs[:ntrain]
validation = pairs[ntrain:nvalidation]
test = pairs[nvalidation:]


# Copy and distribute images and labels (val, train, test)
for pairs, folder in zip((training, validation, test), DESTINATION_DIRS):
    for image_path, label_path in pairs:

        # Create new basenames
        new_label_name = rename_label(label_path)

        print(new_label_name)
        shutil.copy(image_path, DATA_DIR / "images" / folder)
        shutil.copy(label_path, DATA_DIR / "labels" / folder / new_label_name)
