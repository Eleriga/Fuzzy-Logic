import os
import numpy as np
from pathlib import Path
from src.processing.image_loader import ImageLoader

def run_preprocessing_pipeline():

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    loader = ImageLoader()
    
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []

    for ext in extensions:
        image_files.extend(list(raw_dir.glob(ext)))

    for file_path in image_files:
        try:
            print(f"Elaborating: {file_path.name}...")
            
            img = loader.load(file_path, as_gray=True)
            
            img_normalized, original_shape = loader.preprocess(img, normalize=True, flatten=False)
            
            save_name = file_path.stem + ".npy"
            save_path = processed_dir / save_name
            
            np.save(save_path, img_normalized)
            print(f" -> Saved in: {save_path}")
            
        except Exception as e:
            print(f"Error in {file_path.name}: {e}")

if __name__ == "__main__":
    run_preprocessing_pipeline()