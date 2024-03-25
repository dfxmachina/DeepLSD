import glob
from uuid import uuid4

import cv2
from fire import Fire
from tqdm import tqdm

import os
import joblib as jl


def main(image_dir, output_dir):
    out_paths = []
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(image_dir + '**/*.png', recursive=True) + glob.glob(image_dir + '**/*.jpg', recursive=True)

    def _process(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image {image_path}.")
            return []

        tile_size = 768
        h, w, _ = img.shape

        h_offset = (h % tile_size) // 2
        w_offset = (w % tile_size) // 2

        img = img[h_offset:h - h_offset, w_offset:w - w_offset, :]
        paths = []
        # split the image into tiles
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                out_path = os.path.join(output_dir, f"{uuid4()}.png")
                paths.append(out_path)
                crop = img[i:i + tile_size, j:j + tile_size, :]
                if crop.shape[0] < tile_size // 2 or crop.shape[1] < tile_size // 2:
                    continue
                cv2.imwrite(out_path, crop)
        return paths

    jobs = (jl.delayed(_process)(image_path) for image_path in tqdm(image_paths))
    pool = jl.Parallel(n_jobs=-1, backend='threading')
    for paths in pool(jobs):
        out_paths += paths

    with open(f"{output_dir}/output_paths.txt", "w") as f:
        for out_path in out_paths:
            f.write(out_path + "\n")

if __name__ == "__main__":
    Fire(main)
