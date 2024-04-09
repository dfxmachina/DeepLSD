import multiprocessing
import os
from logging import getLogger

import coloredlogs
import cv2
import kornia.feature as KF
import numpy as np
import torch
import torch.nn.functional as F
from fire import Fire
from omegaconf import OmegaConf
from pytorch_metric_learning import distances, losses, miners
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deeplsd.datasets import get_dataset
from deeplsd.datasets.utils.homographies import sample_homography
from deeplsd.models import get_model
from deeplsd.models.deeplsd import DeepLSD, get_group_conv

logger = getLogger(__name__)
coloredlogs.install(level="INFO")

torch.backends.cudnn.benchmark = True


def transform_lines(lines, H):
    """Transform line segments by a homography matrix H."""
    reshaped_lines = lines.reshape(-1, 2)
    reshaped_lines = np.concatenate(
        [reshaped_lines, np.ones((reshaped_lines.shape[0], 1))], axis=1
    )
    transformed_lines = H.dot(reshaped_lines.T).T
    transformed_lines = transformed_lines[:, :2] / transformed_lines[:, 2:]
    return transformed_lines.reshape(-1, 2, 2)


def batch_to_lines(batch, model, max_lines=1024):
    images = batch["image"]
    dfs = batch["df"]
    line_level = batch["line_level"]
    alt_images = (batch["alt_image"].cpu().numpy() * 255).astype(np.uint8)

    lines = []
    np_img = (images.cpu().numpy()[:, 0] * 255).astype(np.uint8)
    np_df = dfs.cpu().numpy()
    np_ll = line_level.cpu().numpy()
    for img, df, ll in zip(np_img, np_df, np_ll):
        line, _, _ = model.detect_afm_lines(img, df, ll, merge=True, filtering=False)
        mask = np.linalg.norm(line[:, 0] - line[:, 1], axis=1) > 30
        line = line[mask]
        line = line[:max_lines]
        lines.append(line)

    result = []

    for i in range(len(images)):
        img = np_img[i]
        alt_img = alt_images[i][0, :, :]
        homo = sample_homography(
            img.shape,
            perspective=True,
            scaling=False,
            rotation=True,
            translation=True,
            patch_ratio=0.8,
            allow_artifacts=True,
        )

        line = lines[i]

        # warp lines and image
        h, w = img.shape
        warped = cv2.warpPerspective(
            alt_img,
            homo,
            (w, h),
            flags=cv2.INTER_LINEAR,
        )
        _, homo_inv = cv2.invert(homo)

        w_lines = transform_lines(line, homo)

        result.append((img, warped, line, w_lines))
    return result


def lines_to_points(lines: np.ndarray):
    start = lines[:, 0, :]
    end = lines[:, 1, :]
    center = (start + end) / 2
    second = (start + center) / 2
    fourth = (center + end) / 2
    points = torch.from_numpy(np.array([start, second, center, fourth, end])).float().view(-1, 2)
    ids = torch.arange(lines.shape[0]).repeat(5)
    return points, ids


def get_descriptors(feature_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Extract the descriptors at specified (fractional) coordinates from a feature map using bilinear interpolation.

    Args:
    - feature_map: The input feature map with shape [C, H, W].
    - coords: The fractional coordinates from which to extract descriptors, with shape [N, 2].

    Returns:
    - Descriptors: Extracted descriptors with shape [N, C], where N is the number of coordinates.
    """
    # Expand feature map from C x H x W to 1 x C x H x W for grid_sample
    feature_map = feature_map.unsqueeze(0)

    # Normalize coords to [-1, 1], as required by F.grid_sample
    _, H, W = feature_map.shape[-3:]
    coords = coords * 2
    coords[:, 0] = coords[:, 0] / (H - 1) - 1
    coords[:, 1] = coords[:, 1] / (W - 1) - 1
    coords = coords.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, N, 2]

    # Apply grid_sample for bilinear interpolation
    descriptors = F.grid_sample(
        feature_map, coords, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # Remove unnecessary dimensions [1, C, 1, N] -> [N, C]
    descriptors = descriptors.squeeze(0).squeeze(1).transpose(0, 1)

    return descriptors


class DescriptorModel(torch.nn.Module):
    def __init__(
            self,
            backbone,
            in_channels: int = 64,
            out_channels: int = 256,
            intermediate_channels=128,
            group_size: int = 16,
    ):
        super().__init__()
        self.pixel_head = torch.nn.Sequential(
            get_group_conv(
                1, intermediate_channels, kernel_size=3, groups=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(intermediate_channels),
            get_group_conv(
                intermediate_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=intermediate_channels // group_size,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(in_channels),
            get_group_conv(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels // group_size,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            get_group_conv(
                in_channels,
                in_channels * 4,
                kernel_size=3,
                padding=1,
                groups=in_channels // group_size,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(in_channels * 4),
            torch.nn.PixelShuffle(2),
        )
        self.feature_head = torch.nn.Sequential(
            get_group_conv(
                intermediate_channels,
                intermediate_channels,
                kernel_size=3,
                padding=1,
                groups=intermediate_channels // group_size,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(intermediate_channels),
            get_group_conv(
                intermediate_channels,
                intermediate_channels,
                kernel_size=3,
                padding=1,
                groups=intermediate_channels // group_size,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(intermediate_channels),
            get_group_conv(
                intermediate_channels,
                out_channels,
                kernel_size=1,
                groups=intermediate_channels // group_size,
            ),
        )
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_features = self.pixel_head(x)
        backbone_features = self.backbone(x)
        return self.feature_head(torch.cat([backbone_features, pixel_features], dim=1))


class Trainer:
    def __init__(
            self,
            config_path,
            checkpoint_path=None,
            output_dir="/tmp/debug/",
            total_epochs=10,
    ):
        self.conf = OmegaConf.load(config_path)
        self.conf.data.update({"double_aug": True, "homographic_augmentation": False})
        self.total_epochs = total_epochs

        if checkpoint_path:
            init_cp = torch.load(str(checkpoint_path), map_location="cpu")
        else:
            init_cp = None

        OmegaConf.set_struct(self.conf, True)

        self.writer = SummaryWriter(log_dir=str(output_dir))
        self.output_dir = output_dir

        dataset = get_dataset(self.conf.data.name)(self.conf.data)
        self.train_loader = dataset.get_data_loader("train")
        self.val_loader = dataset.get_data_loader("val")

        logger.info("Training loader has {} batches".format(len(self.train_loader)))
        logger.info("Validation loader has {} batches".format(len(self.val_loader)))

        self.deep_lsd = get_model(self.conf.model.name)(
            self.conf.model
        )  # type: DeepLSD

        if init_cp is not None:
            if list(init_cp["model"].keys())[0][:7] == "module.":
                # Remove 'module.' added by the DataParallel training
                init_state_dict = {k[7:]: v for k, v in init_cp["model"].items()}
            else:
                init_state_dict = init_cp["model"]
            self.deep_lsd.load_state_dict(init_state_dict)

        # Remove extra heads to save CUDA memory
        self.deep_lsd.angle_head = torch.nn.Identity()
        self.deep_lsd.df_head = torch.nn.Identity()
        logger.info(f"Model: \n{self.deep_lsd}")

        self.model = DescriptorModel(self.deep_lsd.backbone).cuda()

        distance = distances.CosineSimilarity()
        self.mining_func = miners.TripletMarginMiner(
            margin=0.25, distance=distance, type_of_triplets="semihard"
        )
        loss_func = losses.TripletMarginLoss().cuda()
        self.base_loss_func = loss_func
        self.loss_func = losses.CrossBatchMemory(embedding_size=256,
                                                 memory_size=4096,
                                                 miner=self.mining_func,
                                                 loss=self.base_loss_func,
                                                 )
        self.scaler = GradScaler()

        params = [v for k, v in self.model.named_parameters() if "backbone" not in k]
        params += list(self.loss_func.parameters())

        self.optimizer = torch.optim.Adam(
            params=params,
            lr=3e-4,
        )
        self.max_lines_per_image = 1024
        self.sold = KF.SOLD2(pretrained=True).cuda()

    @staticmethod
    def get_valid_coords(coords_a, coords_b, height, width):
        coords_a_valid = (
                (coords_a[:, 0] >= 0) & (coords_a[:, 0] < height) & (coords_a[:, 1] >= 0) & (coords_a[:, 1] < width)
        )
        coords_b_valid = (
                (coords_b[:, 0] >= 0) & (coords_b[:, 0] < height) & (coords_b[:, 1] >= 0) & (coords_b[:, 1] < width)
        )
        valid = coords_a_valid & coords_b_valid
        return valid

    def batch_to_descriptors(self, batch, epoch=0, batch_id=0, prefix="train"):
        images_a, images_b, lines_a, lines_b = zip(
            *batch_to_lines(batch, self.deep_lsd, max_lines=self.max_lines_per_image)
        )

        with autocast():
            feature_map_a = self.model(batch["image"].cuda())
            feature_map_b = self.model(batch["alt_image"].cuda())

        descriptors_a_batch, descriptors_b_batch = [], []
        labels_batch = []

        viz = batch_id == 0 and prefix == "train"
        # FixMe: viz for val?

        id_offset = batch_id * len(batch["image"]) * self.max_lines_per_image
        for i in range(len(images_a)):
            fmap_a = feature_map_a[i]
            fmap_b = feature_map_b[i]

            with autocast():
                keypoints_a, ids = lines_to_points(lines_a[i])
                keypoints_b, _ = lines_to_points(lines_b[i])
                valid = self.get_valid_coords(keypoints_a, keypoints_b, images_a[i].shape[0], images_a[i].shape[1])
                if not valid.any():
                    continue
                keypoints_a = keypoints_a[valid]
                keypoints_b = keypoints_b[valid]
                ids = ids[valid]

                descriptors_a = get_descriptors(fmap_a, keypoints_a.cuda())
                descriptors_b = get_descriptors(fmap_b, keypoints_b.cuda())

                if viz:
                    self.visualize_matches(
                        images_a[i],
                        images_b[i],
                        torch.cat([descriptors_a, descriptors_b]),
                        torch.cat([ids, ids]),
                        keypoints_a,
                        keypoints_b,
                        epoch=epoch,
                    )

                sold_acc = self.get_sold_metrics(
                    batch["image"][i].unsqueeze(1), batch["alt_image"][i].unsqueeze(1), keypoints_a, keypoints_b,
                    torch.cat([ids, ids])
                )

                descriptors_a_batch.append(descriptors_a)
                descriptors_b_batch.append(descriptors_b)
                id_offset += len(ids)
                labels_batch.append(ids + id_offset)

        if len(descriptors_a_batch):
            descriptors = torch.cat(descriptors_a_batch + descriptors_b_batch)
            labels = torch.cat(labels_batch * 2)
        else:
            descriptors = torch.tensor([]).cuda()
            labels = torch.tensor([]).cuda()
        assert len(descriptors) == len(labels)

        if viz:
            for i in range(len(images_a)):
                self.visualize_batch(
                    images_a[i], images_b[i], lines_a[i], lines_b[i], epoch * len(batch["image"]) + i
                )

        return descriptors, labels

    def get_matches(self, descriptors, labels, greedy=True):
        n = len(labels) // 2
        labels_a = labels[:n].cuda()
        descriptors_a = descriptors[:n].cuda()
        descriptors_b = descriptors[n:].cuda()

        distance = distances.CosineSimilarity()

        # Compute cosine similarity between descriptors
        with torch.no_grad():
            with autocast():
                if greedy:
                    similarities = distance(descriptors_a, descriptors_b)
                    costs = -similarities.cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(costs)
                    pred_labels = labels_a[col_ind]
                else:
                    similarities = distance(descriptors_a, descriptors_b)
                    pred_idx = torch.argmax(similarities, dim=1)
                    pred_labels = labels_a[pred_idx]
                return pred_labels

    def get_matching_metrics(self, descriptors, labels):
        n = len(labels) // 2
        labels_b = labels[n:].cuda()
        pred_labels = self.get_matches(descriptors, labels)
        accuracy = (pred_labels == labels_b).float().mean()
        # FixMe: unify how logging is done
        logger.info(f"Accuracy: {accuracy.item():.3f}")
        return accuracy.item()

    def visualize_batch(self, img, warped, lines, warped_lines, epoch=0):
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]
        for j, (l1, l2) in enumerate(zip(lines, warped_lines)):
            color = colors[j % len(colors)]

            (x1, y1), (x2, y2) = l1.round().astype(int)

            cv2.line(img_color, (x1, y1), (x2, y2), color, 4)
            cv2.circle(img_color, (x1, y1), 6, color, -1)
            cv2.circle(img_color, (x2, y2), 6, color, -1)

            (x1, y1), (x2, y2) = l2.round().astype(int)
            cv2.line(warped_color, (x1, y1), (x2, y2), color, 4)
            cv2.circle(warped_color, (x1, y1), 6, color, -1)
            cv2.circle(warped_color, (x2, y2), 6, color, -1)

        self.writer.add_image("image", img_color, epoch, dataformats="HWC")
        self.writer.add_image("warped", warped_color, epoch, dataformats="HWC")

    def visualize_matches(self, img, warped, descriptors, labels, keypoints_a, keypoints_b, epoch=0):
        # FixMe: restore
        return
        matches = self.get_matches(descriptors, labels)
        h, w = img.shape
        joined = cv2.cvtColor(np.hstack([img, warped]), cv2.COLOR_GRAY2BGR).copy()

        for i, (ka, kb, m) in enumerate(zip(keypoints_a, keypoints_b, matches)):
            color = (0, 255, 0) if m == labels[i] else (255, 0, 0)
            x1, y1 = ka.round().numpy().astype(int)
            x2, y2 = (kb.round().numpy().astype(int))
            cv2.circle(joined, (x1, y1), 5, color, -1)
            cv2.circle(joined, (x2 + w, y2), 5, color, -1)
            cv2.line(joined, (x1, y1), (x2 + w, y2), color, 1)

        self.writer.add_image("matches", joined, epoch, dataformats="HWC")

    def get_sold_metrics(self, images_a, images_b, coords_a, coords_b, labels):
        with torch.no_grad():
            with autocast():
                sold_a = self.sold(images_a.cuda())["dense_desc"]
                sold_b = self.sold(images_b.cuda())["dense_desc"]
                # upsampling
                feature_map_a = torch.nn.functional.interpolate(sold_a, size=(images_a.shape[2], images_a.shape[3]),
                                                                mode='bilinear', align_corners=False)
                feature_map_b = torch.nn.functional.interpolate(sold_b, size=(images_b.shape[2], images_b.shape[3]),
                                                                mode='bilinear', align_corners=False)

                descriptors_a = get_descriptors(feature_map_a.squeeze(0), coords_a.cuda())
                descriptors_b = get_descriptors(feature_map_b.squeeze(0), coords_b.cuda())

        acc = self.get_matching_metrics(torch.cat([descriptors_a, descriptors_b]), labels)
        logger.info(f"SOLD accuracy: {acc:.3f}")
        return acc

    def train_epoch(self, epoch):
        train_losses, train_metrics, train_samples = [], [], []
        for batch_id, batch in enumerate(iter(self.train_loader)):
            descriptors, labels = self.batch_to_descriptors(
                batch, epoch=epoch, batch_id=batch_id
            )
            if not len(descriptors):
                continue

            acc = self.get_matching_metrics(descriptors, labels)
            train_metrics.append(acc)
            train_samples.append(len(labels) // 2)
            with autocast():
                self.optimizer.zero_grad()
                indices_tuple = self.mining_func(descriptors, labels)

                loss = self.loss_func(descriptors, labels, indices_tuple)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_losses.append(loss.item())

            if epoch == self.total_epochs // 2:
                # FixMe: reduce learning rate using scheduler
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = 5e-5

        loss = np.mean(train_losses)
        self.writer.add_scalar("train_loss", loss, epoch)

        weighted_acc = np.average(train_metrics, weights=train_samples)
        self.writer.add_scalar("train_weighted_acc", weighted_acc, epoch)

        self.loss_func.reset_queue()

        logger.info(f"Training loss: {loss:.3f}, weighted accuracy {weighted_acc:.3f}, epoch: {epoch}")

    def val_epoch(self, epoch):
        val_losses, val_metrics, val_samples = [], [], []

        for batch_id, batch in enumerate(iter(self.val_loader)):
            descriptors, labels = self.batch_to_descriptors(batch, epoch=epoch, batch_id=batch_id, prefix="val")
            if not len(descriptors):
                continue

            acc = self.get_matching_metrics(descriptors, labels)
            val_metrics.append(acc)
            val_samples.append(len(labels) // 2)
            with autocast():
                indices_tuple = self.mining_func(descriptors, labels)
                loss = self.base_loss_func(descriptors, labels, indices_tuple).item()
                val_losses.append(loss)

        if val_losses:
            loss = np.mean(val_losses)
            self.writer.add_scalar("val_loss", loss.item(), epoch)
            weighted_acc = np.average(val_metrics, weights=val_samples)
            self.writer.add_scalar("val_weighted_acc", weighted_acc, epoch)
            logger.info(
                f"Validation loss: {loss:.3f}, weighted accuracy {weighted_acc:.3f}, epoch: {epoch}")

    def train(self):
        for epoch in tqdm(range(self.total_epochs)):
            self.train_epoch(epoch)
            self.val_epoch(epoch)

    def finish(self):
        self.writer.close()
        state = self.model.state_dict()
        torch.save(state, os.path.join(self.output_dir, "full.pt"))

        torch.save(
            {k: v for k, v in state.items() if "backbone" not in k},
            os.path.join(self.output_dir, "desc_head.pt"),
        )


def main(
        config_path,
        checkpoint_path=None,
        output_dir="/tmp/debug/",
        overfit=False,
        epochs=20,
):
    trainer = Trainer(
        config_path, checkpoint_path, output_dir, overfit, total_epochs=epochs
    )
    trainer.train()
    trainer.finish()


# ToDo: more augs,
# ToDo: more advanced loss / mining
# ToDo: more keypoints per line / more augs per image
# ToDo: draw baseline of sold match

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    Fire(main)
