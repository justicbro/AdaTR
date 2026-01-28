import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorly as tl
import torch
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import TR_func as tr

tl.set_backend("pytorch")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_cores(cores: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [core.clone().detach() for core in cores]


def load_image_tensor(
    image_path: str,
    device: torch.device,
    resize_shape: Optional[Tuple[int, int]] = None,
    resize_width: Optional[int] = None,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if resize_shape is not None:
        image = image.resize(resize_shape, Image.BILINEAR)
    elif resize_width is not None:
        orig_w, orig_h = image.size
        if orig_w == 0:
            raise ValueError(f"Image width is zero for {image_path}")
        scale = resize_width / orig_w
        resize_shape = (resize_width, max(1, int(round(orig_h * scale))))
        image = image.resize(resize_shape, Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    tensor = torch.tensor(array, dtype=torch.float32, device=device)
    return tensor


def collect_image_paths(folder: str, start: int, end: int) -> List[Tuple[int, str]]:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if end < start:
        raise ValueError("end must be >= start")

    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    entries: List[Tuple[int, str]] = []
    for idx in range(start, end + 1):
        found = None
        for ext in exts:
            candidate = os.path.join(folder, f"{idx}{ext}")
            if os.path.isfile(candidate):
                found = candidate
                break
        if found:
            entries.append((idx, found))
        else:
            print(f"[Warning] Image with index {idx} not found; skipping.")
    return entries


def build_synthetic_tensor(
    shape: Sequence[int],
    ranks: Sequence[int],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    set_seed(seed)
    cores = tr.initialize_tr_cores(shape, ranks, device=device.type)
    tensor = tl.tr_to_tensor(cores).to(device=device)
    return tensor


@dataclass
class AlsConfig:
    max_iter: int = 200
    tol: float = 1e-10
    dis_num: int = 50


@dataclass
class GdConfig:
    max_iter: int = 500
    tol: float = 1e-6
    lr: float = 1e-3
    dis_num: int = 50
    optimizer: str = "adam"


def run_single_comparison(
    tensor: torch.Tensor,
    ranks: Sequence[int],
    init_seed: int,
    als_cfg: AlsConfig,
    gd_cfg: GdConfig,
    device: torch.device,
) -> Tuple[float, float]:
    shape = list(tensor.shape)
    set_seed(init_seed)
    init_cores = tr.initialize_tr_cores(shape, ranks, device=device.type)

    als_cores = tr.tr_decompose4(
        tensor,
        clone_cores(init_cores),
        max_iter=als_cfg.max_iter,
        tol=als_cfg.tol,
        dis_num=als_cfg.dis_num,
        device=device.type,
    )
    als_error = torch.norm(tensor - tl.tr_to_tensor(als_cores)).item()

    gd_cores, _ = tr.tr_decompose_gd(
        tensor,
        clone_cores(init_cores),
        max_iter=gd_cfg.max_iter,
        tol=gd_cfg.tol,
        lr=gd_cfg.lr,
        dis_num=gd_cfg.dis_num,
        device=device.type,
        optimizer_type=gd_cfg.optimizer,
    )
    gd_error = torch.norm(tensor - tl.tr_to_tensor(gd_cores)).item()

    return als_error, gd_error


def print_comparison_table(records: List[Tuple[str, float, float]]) -> None:
    header = (
        f"{'Sample':<20}"
        f"{'||X-TN|| ALS':>20}"
        f"{'||X-TN|| GD':>20}"
        f"{'Delta (ALS-GD)':>20}"
    )
    print(header)
    print("-" * len(header))
    for label, als_error, gd_error in records:
        delta = als_error - gd_error
        print(
            f"{label:<20}"
            f"{als_error:>20.6f}"
            f"{gd_error:>20.6f}"
            f"{delta:>20.6f}"
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TR decomposition via ALS (tr_decompose4) and gradient descent."
    )
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "image"],
        default="synthetic",
        help="Use synthetic tensors or real images.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[20, 20, 10],
        help="Shape for synthetic tensors.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[4, 4, 4, 4],
        help="Tensor-ring ranks (len = len(shape) + 1).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of synthetic samples to compare.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=2024,
        help="Base seed for synthetic tensors.",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="./AT_TD/My picture",
        help="Image folder for data-source=image.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start index for image filenames (e.g., 1 -> 1.png).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=1,
        help="End index for image filenames.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize images to specified width/height.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Resize images by width while preserving aspect ratio.",
    )
    parser.add_argument(
        "--als-max-iter",
        type=int,
        default=200,
        help="ALS maximum iterations.",
    )
    parser.add_argument(
        "--als-tol",
        type=float,
        default=1e-10,
        help="ALS tolerance.",
    )
    parser.add_argument(
        "--als-dis-num",
        type=int,
        default=50,
        help="ALS logging interval.",
    )
    parser.add_argument(
        "--gd-max-iter",
        type=int,
        default=500,
        help="Gradient descent maximum iterations.",
    )
    parser.add_argument(
        "--gd-tol",
        type=float,
        default=1e-6,
        help="Gradient descent tolerance.",
    )
    parser.add_argument(
        "--gd-lr",
        type=float,
        default=1e-3,
        help="Gradient descent learning rate.",
    )
    parser.add_argument(
        "--gd-dis-num",
        type=int,
        default=50,
        help="Gradient descent logging interval.",
    )
    parser.add_argument(
        "--gd-optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Gradient descent optimiser type.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=123,
        help="Seed used to generate initial TR cores (per sample).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda or cpu). Defaults to cuda if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_str = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    als_cfg = AlsConfig(
        max_iter=args.als_max_iter,
        tol=args.als_tol,
        dis_num=args.als_dis_num,
    )
    gd_cfg = GdConfig(
        max_iter=args.gd_max_iter,
        tol=args.gd_tol,
        lr=args.gd_lr,
        dis_num=args.gd_dis_num,
        optimizer=args.gd_optimizer,
    )

    records: List[Tuple[str, float, float]] = []

    if args.data_source == "synthetic":
        shape = args.shape
        ranks = args.ranks
        if len(ranks) != len(shape) + 1:
            raise ValueError("`ranks` length must equal len(shape) + 1.")
        for sample_idx in range(args.num_samples):
            tensor_seed = args.data_seed + sample_idx
            tensor = build_synthetic_tensor(shape, ranks, tensor_seed, device)
            init_seed = args.init_seed + sample_idx
            als_error, gd_error = run_single_comparison(
                tensor,
                ranks,
                init_seed,
                als_cfg,
                gd_cfg,
                device,
            )
            label = f"Synthetic-{sample_idx + 1}"
            records.append((label, als_error, gd_error))
    else:
        resize_shape = tuple(args.resize) if args.resize else None
        resize_width = args.resize_width if resize_shape is None else None
        entries = collect_image_paths(args.input_folder, args.start_index, args.end_index)
        if not entries:
            print("No images found for the specified range.")
            return
        ranks = args.ranks
        for idx, image_path in entries:
            tensor = load_image_tensor(
                image_path,
                device=device,
                resize_shape=resize_shape,
                resize_width=resize_width,
            )
            dims = list(tensor.shape)
            if len(ranks) != len(dims) + 1:
                raise ValueError(
                    f"`ranks` length must equal len(image tensor shape) + 1 "
                    f"(got ranks len {len(ranks)}, tensor dims {dims})."
                )
            init_seed = args.init_seed + idx
            als_error, gd_error = run_single_comparison(
                tensor,
                ranks,
                init_seed,
                als_cfg,
                gd_cfg,
                device,
            )
            label = f"Image-{idx}"
            records.append((label, als_error, gd_error))

    print_comparison_table(records)


if __name__ == "__main__":
    main()
