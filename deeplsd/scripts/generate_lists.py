import os

from fire import Fire
from tqdm import tqdm
import shutil


def main(images_dir, gt_dir, new_images_dir=None, new_gt_dir=None):
    images = set(os.listdir(images_dir))
    gts = os.listdir(gt_dir)

    processed = []

    for gt in tqdm(gts):
        img = gt.replace('.hdf5', '.png')
        if img not in images:
            continue
        else:
            processed.append(img)

    num = len(processed)

    # split the data 90/10 to train and val
    split = int(num * 0.9)
    with open(os.path.join(images_dir, 'train.txt'), 'w') as f:
        for img in processed[:split]:
            f.write(img + '\n')
    with open(os.path.join(images_dir, 'val.txt'), 'w') as f:
        for img in processed[split:]:
            f.write(img + '\n')

    if new_images_dir is not None:
        os.makedirs(new_images_dir, exist_ok=True)
        for img in processed:
            shutil.copy(os.path.join(images_dir, img), os.path.join(new_images_dir, img))

    if new_gt_dir is not None:
        os.makedirs(new_gt_dir, exist_ok=True)
        for gt in processed:
            gt = gt.replace('.png', '.hdf5')
            shutil.copy(os.path.join(gt_dir, gt), os.path.join(new_gt_dir, gt))


if __name__ == "__main__":
    Fire(main)
