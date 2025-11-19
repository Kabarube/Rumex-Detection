from pathlib import Path
import shutil
import glob
import random

filetype = "*.jpg"

# Collect paths for all the jpg's
all_jpgs = glob.glob("./DJI_202508171710_007_Create-Area-Route31/*.jpg")

# Select 100 random images from the list
selected_images = random.sample(all_jpgs, 100)
destination = Path('./selection')
destination.mkdir(exist_ok=True)

for img in selected_images:
    img_path = Path(img)
    shutil.copy2(img_path, destination / img_path.name)
