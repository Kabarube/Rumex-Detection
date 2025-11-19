'''
 # @ Author: Kai Ruben Enerhaugen
 # @ Email: kai.r.h.enerhaugen@nmbu.no
 # @ Create Time: 2025-11-18
 # @ Modified time: 2025-11-19
 '''
from ultralytics import YOLO
from classes import DataHandler
from pathlib import Path

OUTPUT_DIR = Path("./data")
IMAGE_PATH = Path("./images")
LABEL_PATH = Path("./labels")
IMAGE_EXTENSION = ".jpg"

if __name__ == "__main__":
    handler = DataHandler(
        source_dir=IMAGE_PATH,
        output_dir=OUTPUT_DIR,
        image_filetype=IMAGE_EXTENSION,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    handler.process_data(
        labels_dir=LABEL_PATH,
        shuffle=True
    )



    # model = YOLO("yolo11n.pt")
    # model.train(data="rumex.yaml", epochs=100, imgsz=640, batch=8, device="0")


