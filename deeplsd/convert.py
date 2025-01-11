from pathlib import Path
from time import time
from typing import Sequence, Union

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from fire import Fire
from onnxruntime_tools import optimizer
from onnxsim import simplify
from pytlsd import lsd

from deeplsd.predictor import Predictor


def optimize_onnx_model(input_file, output_file, opt_level=2):
    """
    We apply standard optimizations from onnxruntime_tools, and remove unnecessary initializers to avoid warnings.
    Heavily inspired by https://github.com/microsoft/onnxruntime/blob/master/tools/python/remove_initializer_from_input.py
    """
    optimizer.optimize_by_onnxruntime(onnx_model_path=input_file, optimized_model_path=output_file, opt_level=opt_level)
    model = onnx.load(output_file)
    if model.ir_version < 4:
        raise RuntimeError("Model with ir_version below 4 requires to include initializer in graph input")

    inputs = model.graph.input
    name_to_input = {x.name: x for x in inputs}
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, output_file)


def _get_dummy_input(img_size: Union[int, Sequence]):
    if isinstance(img_size, int):
        return torch.rand(1, 1, img_size, img_size) * 255
    return torch.rand(1, *img_size) * 255


class WrappedModel(torch.nn.Module):
    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, batch: torch.Tensor):
        # for ONNX conversion only
        base = self.core.backbone(batch / 255.)
        df_norm = self.core.df_head(base).squeeze(1)
        df = self.core.denormalize_df(df_norm)
        line_level = self.core.angle_head(base).squeeze(1) * np.pi
        gradnorm = torch.maximum(5 - df, torch.zeros_like(df)).double()
        angle = line_level - np.pi / 2
        return gradnorm, angle


def save_to_onnx(
        model: torch.nn.Module,
        out_name: str,
        img_size: Union[int, Sequence] = 64,
        train_mode=False,
        opset_version=12,
        opt_level=2,
        simplify_model=True,
):
    model = model.train(mode=train_mode).cpu()
    img = _get_dummy_input(img_size)

    input_names = ["input"]
    output_names = ["gradnorm", 'angle']
    dyn_axes = {0: "batch_size", 2: "height", 3: "width"}
    torch.onnx.export(
        model,
        (img,),
        out_name,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes={"input": dyn_axes, "gradnorm": dyn_axes, 'angle': dyn_axes},
    )
    if opt_level is not None:
        optimize_onnx_model(out_name, out_name, opt_level)

    if simplify_model:
        model_to_simplify = onnx.load(out_name)
        model_simp, check = simplify(
            model_to_simplify,
            input_data={"input": img.numpy()},
            dynamic_input_shape=True,
            perform_optimization=False,
        )
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, out_name)

    return out_name


def compute_image_grad(img, ksize=7):
    blur_img = cv2.GaussianBlur(img, (ksize, ksize), 1).astype(np.float32)
    dx = np.zeros_like(blur_img)
    dy = np.zeros_like(blur_img)
    dx[:, 1:] = (blur_img[:, 1:] - blur_img[:, :-1]) / 2
    dx[1:, 1:] = dx[:-1, 1:] + dx[1:, 1:]
    dy[1:] = (blur_img[1:] - blur_img[:-1]) / 2
    dy[1:, 1:] = dy[1:, :-1] + dy[1:, 1:]
    gradangle = np.arctan2(dy, dx)
    return gradangle


def compute_image_grad_torch(img, ksize=7):
    # Add batch and channel dimensions if needed
    if len(img.shape) == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(img.shape) == 3:
        img = img.unsqueeze(0)

    # Gaussian blur
    sigma = 1.0
    kernel_size = ksize
    # Create Gaussian kernel
    x_coord = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x_coord, x_coord)
    gaussian_kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # Apply Gaussian blur
    blur_img = torch.nn.functional.conv2d(img, gaussian_kernel.to(img.device), padding=kernel_size // 2)

    # Calculate gradients
    dx = torch.zeros_like(blur_img)
    dy = torch.zeros_like(blur_img)

    # Horizontal gradient
    dx[:, :, :, 1:] = (blur_img[:, :, :, 1:] - blur_img[:, :, :, :-1]) / 2
    dx[:, :, 1:, 1:] = dx[:, :, :-1, 1:] + dx[:, :, 1:, 1:]

    # Vertical gradient
    dy[:, :, 1:] = (blur_img[:, :, 1:] - blur_img[:, :, :-1]) / 2
    dy[:, :, 1:, 1:] = dy[:, :, 1:, :-1] + dy[:, :, 1:, 1:]

    # Calculate gradient angle
    gradangle = torch.atan2(dy, dx)
    # Remove extra dimensions if they were added
    if len(img.shape) == 4 and img.shape[0] == 1 and img.shape[1] == 1:
        gradangle = gradangle.squeeze()

    return gradangle


def align_with_grad_angle(angle, img):
    """ Starting from an angle in [0, pi], find the sign of the angle based on
        the image gradient of the corresponding pixel. """
    # Image gradient
    # img_grad_angle = compute_image_grad(img)
    img_grad_angle = compute_image_grad_torch(torch.from_numpy(img), ksize=7).numpy()

    # Compute the distance of the image gradient to the angle
    # and angle - pi
    pred_grad = np.mod(angle, np.pi)  # in [0, pi]
    pos_dist = np.minimum(np.abs(img_grad_angle - pred_grad),
                          2 * np.pi - np.abs(img_grad_angle - pred_grad))
    neg_dist = np.minimum(
        np.abs(img_grad_angle - pred_grad + np.pi),
        2 * np.pi - np.abs(img_grad_angle - pred_grad + np.pi))

    # Assign the new grad angle to the closest of the two
    is_pos_closest = np.argmin(np.stack([neg_dist, pos_dist],
                                        axis=-1), axis=-1).astype(bool)
    new_grad_angle = np.where(is_pos_closest, pred_grad, pred_grad - np.pi)
    return new_grad_angle, img_grad_angle


def preprocess_angle(angle, img):
    oriented_grad_angle, img_grad_angle = align_with_grad_angle(angle, img)
    oriented_grad_angle = np.mod(oriented_grad_angle - np.pi / 2, 2 * np.pi)
    oriented_grad_angle[0] = -1024
    oriented_grad_angle[:, 0] = -1024
    return oriented_grad_angle.astype(np.float64)


def prepare_lsd_inputs(gradnorm, angle, img):
    angle = preprocess_angle(angle, img)
    angle[gradnorm < 3] = -1024
    return angle


class ONNXDetector:
    def __init__(self, onnx_path: str):
        self.ort_session = ort.InferenceSession(onnx_path)

    def __call__(self, img: np.ndarray, grad_nfa=True):
        img = np.expand_dims(img, (0, 1)).astype(np.float32)
        ort_inputs = {"input": img}
        gradnorm, angle = self.ort_session.run(None, ort_inputs)
        gradnorm = gradnorm.squeeze(0)
        angle = angle.squeeze(0)
        img = img.squeeze(0).squeeze(0)
        angle = prepare_lsd_inputs(gradnorm, angle, img)
        lines = lsd(img.astype(np.float64), scale=1., gradnorm=gradnorm,
                    gradangle=angle, grad_nfa=grad_nfa)[:, :4].reshape(-1, 2, 2)
        return lines


def main(path_to_weights: str, output_path: str, img_size: int = 256, img_path: str = "path/to/img.jpg"):
    path_to_weights = Path(path_to_weights)
    predictor = Predictor(ckpt=path_to_weights)

    predictor.set_custom_params({'grad_thresh': 3.0, "filtering": False})
    # the higher grad_thresh, the more angles are effectively zeroes out
    predictor.set_custom_params({'line_neighborhood': 5}, key=None)

    model = WrappedModel(predictor.net)
    onnx_path = save_to_onnx(model, output_path, img_size=img_size // 2)
    inputs = _get_dummy_input(img_size)

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": inputs.numpy()}
    t = time()
    distance_ort, line_level_ort = ort_session.run(None, ort_inputs)
    print(f"ONNX inference time: {time() - t:.3f}s")

    t = time()
    with torch.no_grad():
        distance_torch, line_level_torch = model(inputs)
        distance_torch = distance_torch.numpy()
        line_level_torch = line_level_torch.numpy()
        print(f"Torch inference time: {time() - t:.3f}s")

    np.testing.assert_allclose(distance_ort, distance_torch, rtol=0, atol=1e-4)
    np.testing.assert_allclose(line_level_ort, line_level_torch, rtol=0, atol=1e-2)  # 0.57 degrees, ok-ish

    detector = ONNXDetector(onnx_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    t = time()
    lines_onnx = detector(img)
    print(f"Detected {len(lines_onnx)} lines in {time() - t:.3f}s")
    t = time()
    lines_base = predictor.predict(img)
    print(f"Detected {len(lines_base)} lines in {time() - t:.3f}s")

    # grad_angle_base = compute_image_grad(img)
    # grad_angle_toch = compute_image_grad_torch(torch.tensor(img), ksize=7).numpy()
    #
    # np.testing.assert_allclose(grad_angle_base, grad_angle_toch, rtol=0, atol=1e-2)
    # breakpoint()

    # img_a = cv2.imread(img_path)
    # img_b = img_a.copy()
    #
    # img_a = predictor.draw_lines(img_a, lines_base.round().astype(int))
    # img_b = predictor.draw_lines(img_b, lines_onnx.round().astype(int))
    #
    # cv2.imshow("Base", np.hstack((img_a, img_b)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    Fire(main)
