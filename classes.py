'''
 # @ Author: Kai Ruben Enerhaugen
 # @ Email: kai.r.h.enerhaugen@nmbu.no
 # @ Create Time: 2025-11-19
 # @ Modified time: 2025-11-19
 '''

from pathlib import Path
import shutil
import re
import random
from typing import List, Tuple

"""Data handling and processing for image and label datasets."""

class DataHandler:

    def __init__(
            self, 
            source_dir: Path,
            output_dir: Path,
            image_filetype: str = '.jpg',
            train_ratio: float = 0.6,
            val_ratio: float = 0.2,
            test_ratio: float = 0.2):
        
        """Initialize DataHandler with directory paths and ratios."""
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.image_filetype = image_filetype
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self._validate_ratios()

    def _validate_ratios(self) -> None:
        """Ensure that the sum of ratios equals 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) >= 1e-6:
            raise ValueError("Ratios must sum to 1!")
        
    def validate_output_dir(self) -> None:
        """Check if output directory is empty."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise FileExistsError("Output directory already exists!")

    def create_directories(self) -> None:
        """Create necessary directories for images and labels."""
        for folder in ["images", "labels"]:
            for subfolder in ['train', 'val', 'test']:
                (self.output_dir / folder / subfolder).mkdir(parents=True, exist_ok=True)

    def get_images(self) -> List[Path]:
        """Retrieve all image files from the source directory."""
        return [img for img in self.source_dir.iterdir() if img.is_file() and img.suffix.lower() == self.image_filetype]

    def get_labels(self, labels_dir: Path) -> List[Path]:
        """Retrieve all label files from the specified labels directory."""
        return [lbl for lbl in labels_dir.iterdir() if lbl.is_file() and lbl.suffix.lower() == '.txt']
    
    @staticmethod
    def normalize_label_name(label_path: Path) -> str:
        """Extract normalized label name from the file path."""
        m = re.search(r"(DJI_.*)", label_path.name)
        if not m:
            raise ValueError(f"Unexpected filename format: {label_path.name}")
        return m.group(1)
    
    def match_images_to_labels(
            self,
            images: List[Path],
            labels: List[Path]
            ) -> List[Tuple[Path, Path]]:
        """Match image files to their corresponding label files."""
        pairs = []
        for label in labels:
            m = re.search(r"(\d+_D)$", label.stem)
            if not m:
                continue
            for image in images:
                if image.stem.endswith(m.group(1)):
                    pairs.append((image, label))
        return pairs
    
    def split_data(
            self,
            pairs: List[Tuple[Path, Path]],
            shuffle: bool = True
            ) -> Tuple[List[Tuple[Path, Path]], ...]:
        """Split data into training, validation, and test sets."""
        if shuffle:
            random.shuffle(pairs)

        n_total = len(pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_train + n_total * self.val_ratio)
        train = pairs[:n_train]
        val = pairs[n_train:n_val]
        test = pairs[n_val:]

        return train, val, test
    
    def copy_files(
            self,
            pairs: List[Tuple[Path, Path]],
            split_name: str
            ) -> None:
        """Copy image and label files to their respective directories."""

        for image_path, label_path in pairs:
            # Normalize label filename
            new_label_name = self.normalize_label_name(label_path)

            # Copy files
            shutil.copy(image_path, self.output_dir / "images" / split_name)
            shutil.copy(label_path, self.output_dir / "labels" / split_name / new_label_name)

    def process_data(self, labels_dir: Path, shuffle: bool) -> None:
        """Execute the full data processing pipeline."""

        # Validate and prepare direcories
        self.validate_output_dir()
        self.create_directories()

        images = self.get_images()
        labels = self.get_labels(labels_dir)

        print(f"Found {len(images)} images and {len(labels)} labels.")

        # Match images to labels and split
        pairs = self.match_images_to_labels(images, labels)
        print(f"Matched {len(pairs)} image-label pairs.")

        train, val, test = self.split_data(pairs, shuffle)
        print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test.")

        # Copy files to respective directories
        self.copy_files(train, 'train')
        self.copy_files(val, 'val')
        self.copy_files(test, 'test')
        print("Data processing complete.")