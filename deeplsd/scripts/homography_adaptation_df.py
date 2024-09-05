"""
Run the homography adaptation for all images in a given folder
to regress and aggregate line distance function maps.
"""

import argparse
import os

import cv2
import h5py
import numpy as np
import torch
from afm_op import afm
from joblib import Parallel, delayed
from pytlsd import lsd
from tqdm import tqdm
from skimage import exposure

from ..datasets.utils.data_augmentation import random_contrast
from ..datasets.utils.homographies import sample_homography, warp_lines

homography_params = {
    "translation": True,
    "rotation": True,
    "scaling": True,
    "perspective": True,
    "scaling_amplitude": 0.2,
    "perspective_amplitude_x": 0.2,
    "perspective_amplitude_y": 0.2,
    "patch_ratio": 0.85,
    "max_angle": 1.57,
    "allow_artifacts": True,
}


def pca_gray(img):
    h, w, _ = img.shape
    _img = img.reshape(-1, 3) / 255.0
    _img_mean = _img - np.mean(_img, axis=0)
    covariance_matrix = np.cov(_img_mean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    principal_component = np.abs(eigenvectors[:, np.argmax(eigenvalues)])
    _img_pca = np.dot(_img_mean, principal_component)
    _img_pca = _img_pca.reshape(h, w)
    _img_pca = (_img_pca - _img_pca.min()) / (_img_pca.max() - _img_pca.min()) * 255
    return _img_pca


def equalize_clahe(x):
    return (exposure.equalize_adapthist(x, clip_limit=0.02).astype(np.float32) * 255).astype(np.uint8)


def subtract_blurred(x, ksize=31):
    diff = cv2.subtract(x.astype("float32"), cv2.GaussianBlur(x, (ksize, ksize), 0).astype("float32"))
    return equalize_clahe(np.clip(x + diff, 0, 255).astype(np.uint8))


def ha_df(img, num=100, border_margin=3, min_counts=5, with_tqdm=False):
    h, w = img.shape[:2]
    if img.ndim == 3:
        img = pca_gray(img)

    size = (w, h)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_margin * 2, border_margin * 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pix_loc = torch.stack(torch.meshgrid(torch.arange(h, device=device),
                                         torch.arange(w, device=device), indexing='ij'), dim=-1)

    raster_lines = torch.zeros((h, w), dtype=torch.uint8, device=device)

    df_maps = torch.empty((num, h, w), device=device)
    offsets = torch.empty((num, h, w, 2), device=device)
    counts = torch.empty((num, h, w), device=device)

    for i in tqdm(range(num), disable=not with_tqdm):
        H = np.eye(3) if i == 0 else sample_homography(img.shape, **homography_params)
        H_inv = np.linalg.inv(H)
        warped_img = torch.from_numpy(
            cv2.warpPerspective(img, H, size, borderMode=cv2.BORDER_REPLICATE)).to(device)
        warped_lines = lsd(warped_img.cpu().numpy(), scale=1, sigma_scale=0.4, grad_nfa=False)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        lines = torch.from_numpy(warp_lines(warped_lines, H_inv)).to(device)

        num_lines = len(lines)
        cuda_lines = lines[:, :, [1, 0]].reshape(-1, 4).unsqueeze(0).float()
        offset = afm(cuda_lines, torch.IntTensor([[0, num_lines, h, w]]).cuda(), h, w)[0][0].permute(1, 2, 0)[:, :,
                 [1, 0]]

        df_maps[i] = torch.norm(offset, dim=-1)
        offsets[i] = offset

        counts[i] = torch.from_numpy(
            cv2.erode(cv2.warpPerspective(np.ones_like(img), H_inv, size, flags=cv2.INTER_NEAREST),
                      kernel)).to(device)
        raster_lines += ((df_maps[i] < 1) * counts[i]).to(torch.uint8)

    counts_mask = counts == 0

    avg_df = torch.nanquantile(torch.where(counts_mask, torch.tensor(float('nan'), device=device), df_maps), 0.25,
                               dim=0)
    avg_offset = torch.nanquantile(
        torch.where(counts_mask.unsqueeze(-1), torch.tensor(float('nan'), device=device), offsets), 0.25, dim=0)
    avg_closest = pix_loc + avg_offset

    avg_angle = torch.fmod(torch.atan2(avg_offset[:, :, 0], avg_offset[:, :, 1]) + torch.pi / 2, torch.pi)
    raster_lines = cv2.dilate(np.where(raster_lines.cpu().numpy() > min_counts, 1, 0).astype(np.uint8),
                              np.ones((21, 21), dtype=np.uint8))
    bg_mask = 1 - raster_lines

    return avg_df.cpu().numpy(), avg_angle.cpu().numpy(), avg_closest[:, :, [1, 0]].cpu().numpy(), bg_mask


def process_image(img_path, randomize_contrast, num_H, output_folder):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image {img_path}.")
        return

    if randomize_contrast is not None:
        img = randomize_contrast(img)

    img = subtract_blurred(img)

    # Run homography adaptation
    df, angle, closest, bg_mask = ha_df(img, num=num_H)

    # Save the DF in a hdf5 file
    out_path = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_folder, out_path) + ".hdf5"

    assert (
            df.shape[:2]
            == angle.shape[:2]
            == closest.shape[:2]
            == bg_mask.shape[:2]
            == img.shape[:2]
    )

    with h5py.File(out_path, "w") as f:
        f.create_dataset("df", data=df.flatten())
        f.create_dataset("line_level", data=angle.flatten())
        f.create_dataset("closest", data=closest.flatten())
        f.create_dataset("bg_mask", data=bg_mask.flatten())


def export_ha(images_list, output_folder, num_H=100, rdm_contrast=False, n_jobs=1):
    # Parse the data
    with open(images_list, "r") as f:
        image_files = f.readlines()
    image_files = [path.strip("\n") for path in image_files]
    image_files = [path for path in image_files if os.path.exists(path)]

    # Random contrast object
    print(f"Random contrast: {rdm_contrast}")
    randomize_contrast = random_contrast() if rdm_contrast else None
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in parallel
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_image)(img_path, randomize_contrast, num_H, output_folder)
        for img_path in tqdm(image_files)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images_list", type=str, help="Path to a txt file containing the image paths."
    )
    parser.add_argument("output_folder", type=str, help="Output folder.")
    parser.add_argument(
        "--num_H", type=int, default=100, help="Number of homographies used during HA."
    )
    parser.add_argument(
        "--random_contrast",
        action="store_true",
        help="Add random contrast to the images (disabled by default).",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of jobs to run in parallel."
    )
    args = parser.parse_args()
    export_ha(
        args.images_list,
        args.output_folder,
        args.num_H,
        args.random_contrast,
        args.n_jobs,
    )
