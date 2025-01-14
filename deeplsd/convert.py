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
    optimizer.optimize_by_onnxruntime(
        onnx_model_path=input_file,
        optimized_model_path=output_file,
        opt_level=opt_level,
    )
    model = onnx.load(output_file)
    if model.ir_version < 4:
        raise RuntimeError(
            "Model with ir_version below 4 requires to include initializer in graph input"
        )

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

    def forward(
        self,
        batch: torch.Tensor,
    ):
        base = self.core.backbone(batch / 255.0)
        df_norm = self.core.df_head(base).squeeze(1)
        df = self.core.denormalize_df(df_norm)
        line_level = self.core.angle_head(base).squeeze(1) * torch.pi
        gradnorm = torch.maximum(5 - df, torch.zeros_like(df)).double()
        angle = line_level - torch.pi / 2

        angle = preprocess_angle(angle, batch).squeeze(0)
        angle[gradnorm < 3] = -1024

        return gradnorm, angle


def save_to_onnx(
    model: torch.nn.Module,
    out_name: str,
    img_size: Union[int, Sequence] = 64,
    train_mode=False,
    opset_version=12,
    opt_level=1,
    simplify_model=True,
):
    model = model.train(mode=train_mode).cpu()
    img = _get_dummy_input(img_size)

    input_names = ["input"]
    output_names = ["gradnorm", "angle"]
    dyn_axes = {0: "batch_size", 2: "height", 3: "width"}
    torch.onnx.export(
        model,
        (img,),
        out_name,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes={"input": dyn_axes, "gradnorm": dyn_axes, "angle": dyn_axes},
    )
    if opt_level is not None:
        optimize_onnx_model(out_name, out_name, opt_level)

    if simplify_model:
        model_to_simplify = onnx.load(out_name)
        model_simp, check = simplify(
            model_to_simplify,
            input_data={"input": img.numpy().astype(np.float16)},
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


def get_gaussian_kernel2d(kernel_size: tuple):
    assert len(kernel_size) == 2, "Kernel size must be a tuple of (height, width)."
    assert all(
        k % 2 == 1 for k in kernel_size
    ), "Kernel sizes must be odd for proper centering."

    height, width = kernel_size
    sigma = 1

    y = torch.arange(-(height // 2), height // 2 + 1, dtype=torch.float32)
    x = torch.arange(-(width // 2), width // 2 + 1, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()

    return gaussian


def compute_image_grad_torch(img, ksize=7):
    kernel = get_gaussian_kernel2d((ksize, ksize)).unsqueeze(0).unsqueeze(0)
    blur_img = torch.nn.functional.conv2d(img, kernel, padding=ksize // 2)
    dx = torch.zeros_like(blur_img)
    dy = torch.zeros_like(blur_img)

    dx[:, :, :, 1:] = (blur_img[:, :, :, 1:] - blur_img[:, :, :, :-1]) / 2
    dx[:, :, 1:, 1:] = dx[:, :, :-1, 1:] + dx[:, :, 1:, 1:]

    dy[:, :, 1:] = (blur_img[:, :, 1:] - blur_img[:, :, :-1]) / 2
    dy[:, :, 1:, 1:] = dy[:, :, 1:, :-1] + dy[:, :, 1:, 1:]

    gradangle = torch.atan2(dy, dx)
    return gradangle


def align_with_grad_angle(angle, img):
    img_grad_angle = compute_image_grad_torch(img, ksize=7)

    pred_grad = torch.fmod(angle, torch.pi)  # in [0, pi]
    pos_dist = torch.minimum(
        torch.abs(img_grad_angle - pred_grad),
        2 * np.pi - torch.abs(img_grad_angle - pred_grad),
    )
    neg_dist = torch.minimum(
        torch.abs(img_grad_angle - pred_grad + torch.pi),
        2 * torch.pi - torch.abs(img_grad_angle - pred_grad + torch.pi),
    )

    is_pos_closest = torch.argmin(
        torch.stack([neg_dist, pos_dist], axis=-1), axis=-1
    ).bool()
    new_grad_angle = torch.where(is_pos_closest, pred_grad, pred_grad - torch.pi)
    return new_grad_angle, img_grad_angle


def preprocess_angle(angle, img):
    oriented_grad_angle, img_grad_angle = align_with_grad_angle(angle, img)
    oriented_grad_angle = torch.fmod(oriented_grad_angle - torch.pi / 2, 2 * torch.pi)
    oriented_grad_angle[:, :, 0] = -1024
    oriented_grad_angle[:, :, :, 0] = -1024
    return oriented_grad_angle.double()


def prepare_lsd_inputs(gradnorm, angle, img):
    angle = torch.from_numpy(angle)
    gradnorm = torch.from_numpy(gradnorm)
    img = torch.from_numpy(img)

    angle = preprocess_angle(angle, img)
    angle[gradnorm < 3] = -1024
    return angle


class ONNXDetector:
    def __init__(self, onnx_path: str):
        self.ort_session = ort.InferenceSession(onnx_path)

    def __call__(self, img: np.ndarray, grad_nfa=True):
        img = np.expand_dims(img, (0, 1)).astype(np.float16)
        ort_inputs = {"input": img}
        gradnorm, angle = self.ort_session.run(None, ort_inputs)
        gradnorm = gradnorm.squeeze(0)
        angle = angle.squeeze(0)
        img = img.squeeze(0).squeeze(0)
        lines = lsd(
            img.astype(np.float64),
            scale=1.0,
            gradnorm=gradnorm,
            gradangle=angle,
            grad_nfa=grad_nfa,
        )[:, :4].reshape(-1, 2, 2)
        return lines


def main(
    path_to_weights: str,
    output_path: str,
    img_size: int = 256,
    img_path: str = "path/to/img.jpg",
):
    path_to_weights = Path(path_to_weights)
    predictor = Predictor(ckpt=path_to_weights)

    predictor.set_custom_params({"grad_thresh": 3.0, "filtering": False})
    predictor.set_custom_params({"line_neighborhood": 5}, key=None)

    model = WrappedModel(predictor.net)

    tensor = (
        torch.from_numpy(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    with torch.no_grad():
        t = time()
        gradnorm, angle = model(tensor)

        lines = lsd(
            cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64),
            scale=1.0,
            gradnorm=gradnorm.squeeze(0).numpy(),
            gradangle=angle.squeeze(0).numpy(),
            grad_nfa=True,
        )
        lines = lines[:, :4].reshape(-1, 2, 2)
        print(f"Detected {len(lines)} lines with PyTorch in {time() - t:.3f}s")

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
    np.testing.assert_allclose(
        line_level_ort, line_level_torch, rtol=0, atol=1e-2
    )  # 0.57 degrees, ok-ish

    detector = ONNXDetector(onnx_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    t = time()
    lines_onnx = detector(img)
    print(f"Detected {len(lines_onnx)} lines in {time() - t:.3f}s")
    t = time()
    lines_base = predictor.predict(img)
    print(f"Detected {len(lines_base)} lines in {time() - t:.3f}s")

    img_a = cv2.imread(img_path)
    img_b = img_a.copy()

    img_a = predictor.draw_lines(img_a, lines_base.round().astype(int))
    img_b = predictor.draw_lines(img_b, lines_onnx.round().astype(int))

    cv2.imshow("Base", np.hstack((img_a, img_b)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Fire(main)
