import os
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Literal, Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import unet
import unetFixed
import prof_unet
from SplitNetInterp import SplitNetInterp

from SplitNet import SplitNet


# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------

ModelType = Literal[
    "splitnet_attn",
    "splitnet",
    "unet",
    "attn_unet",
    "unetfixed",
]

DatasetMode = Literal[
    "border",
    "fixed",
]

TrainingMode = Literal[
    "physics_limited",
    "baseline_full",
]


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    model_type: ModelType = "splitnet_attn"
    dataset_mode: DatasetMode = "fixed"
    training_mode: TrainingMode = "physics_limited"

    mse_weight: float = 1.0
    darcy_weight: float = 1.0

    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3

    channels: str = "KP"

    train_sims_path: str = "../train_sims.npy"
    val_sims_path: str = "../val_sims.npy"
    sim_max_exclusive: Optional[int] = 500

    save_dir: str = "minimum_info/results"
    run_name: Optional[str] = None
    save_prefix: Optional[str] = None
    save_best: bool = True

    # If None:
    #   physics_limited -> val_supervised_mse
    #   baseline_full    -> val_total_mse
    best_metric: Optional[str] = None

    dataset_kwargs: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Darcy physics loss
# -----------------------------------------------------------------------------

def darcy_loss_from_output(out: torch.Tensor) -> torch.Tensor:
    """
    Computes mean squared Darcy residual:

        div(K * grad(P))^2

    Expected tensor shape:
        [B, C, H, W]

    Channel order:
        channel 0 = K
        channel 1 = P

    Returns:
        scalar tensor
    """
    if out.shape[1] < 2:
        raise ValueError("Darcy loss requires at least two channels: K and P.")

    k = out[:, 0:1]
    p = out[:, 1:2]

    p_y, p_x = torch.gradient(p, dim=(-2, -1))

    flux_y = k * p_y
    flux_x = k * p_x

    div_y = torch.gradient(flux_y, spacing=(1,), dim=(-2,))[0]
    div_x = torch.gradient(flux_x, spacing=(1,), dim=(-1,))[0]

    residual = div_y + div_x
    return (residual ** 2).mean()


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

def get_num_channels(channels: str) -> int:
    if channels == "all":
        return 3
    if channels == "KP":
        return 2
    if channels in ["K", "P", "phi"]:
        return 1

    raise ValueError("channels must be 'all', 'KP', 'K', 'P', or 'phi'.")


def make_model(model_type: ModelType = "splitnet_attn", channels: str = "KP") -> nn.Module:
    """
    Creates the requested model architecture.
    """
    model_type = model_type.lower()
    num_channels = get_num_channels(channels)

    if model_type == "splitnet_attn":
        if channels != "KP":
            raise ValueError("SplitNet only supports channels='KP'.")
        return SplitNet(attn=True).to(DEVICE)

    if model_type == "splitnet":
        if channels != "KP":
            raise ValueError("SplitNet only supports channels='KP'.")
        return SplitNet(attn=False).to(DEVICE)

    if model_type == "unet":
        return unet.SmallUnet(channels=num_channels).to(DEVICE)

    if model_type == "attn_unet":
        return unet.AttnUnet(channels=num_channels).to(DEVICE)

    if model_type == "unetfixed":
        return unetFixed.UNet(
            in_channels=num_channels,
            num_classes=num_channels
        ).to(DEVICE)
    
    if model_type == "splitnet_interp":
        if channels != "KP":
            raise ValueError("SplitNetInterp only supports channels='KP'.")
        return SplitNetInterp().to(DEVICE)

    raise ValueError(f"Unknown model_type: {model_type}")


# -----------------------------------------------------------------------------
# Dataset selection
# -----------------------------------------------------------------------------

def get_dataset_class(dataset_mode: DatasetMode, training_mode: TrainingMode):

    if training_mode == "physics_limited":
        if dataset_mode == "border":
            return datasets.BorderDenseDatasetLimited
        if dataset_mode == "fixed":
            return datasets.FixedDenseDatasetLimited
        if dataset_mode == "border_sides":
            return datasets.BorderDenseDatasetLimitedSides

    if training_mode == "baseline_full":
        if dataset_mode == "border":
            return datasets.BorderDenseDatasetFull
        if dataset_mode == "fixed":
            return datasets.FixedDenseDatasetFull
        if dataset_mode == "border_sides":
            return datasets.BorderDenseDatasetFullSides

    raise ValueError(
        f"Invalid dataset/training combination: "
        f"dataset_mode={dataset_mode}, training_mode={training_mode}"
    )


def load_sim_ids(path: str, sim_max_exclusive: Optional[int]) -> np.ndarray:
    sims = np.load(path)

    if sim_max_exclusive is not None:
        sims = sims[sims < sim_max_exclusive]

    return sims


class ChannelSelectDataset(torch.utils.data.Dataset):
    """
    Safety wrapper that ensures the dataset only returns requested channels.
    """

    def __init__(self, base_dataset, channels: str = "KP"):
        self.base_dataset = base_dataset
        self.channels = channels

    def _idx(self):
        if self.channels == "all":
            return [0, 1, 2]
        if self.channels == "KP":
            return [0, 1]
        if self.channels == "K":
            return [0]
        if self.channels == "P":
            return [1]
        if self.channels == "phi":
            return [2]

        raise ValueError(f"Bad channels: {self.channels}")

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        chans = self._idx()

        if len(item) == 3:
            feat, label, mask = item
            return feat[chans], label[chans], mask

        feat, label = item
        return feat[chans], label[chans]


def make_loaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    train_sims = load_sim_ids(config.train_sims_path, config.sim_max_exclusive)
    val_sims = load_sim_ids(config.val_sims_path, config.sim_max_exclusive)

    dataset_cls = get_dataset_class(config.dataset_mode, config.training_mode)

    kwargs = dict(config.dataset_kwargs or {})
    kwargs["channels"] = config.channels

    train_data = dataset_cls(train_sims, **kwargs)
    val_data = dataset_cls(val_sims, **kwargs)

    train_data = ChannelSelectDataset(train_data, config.channels)
    val_data = ChannelSelectDataset(val_data, config.channels)

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Saving helpers
# -----------------------------------------------------------------------------

def make_run_prefix(config: ExperimentConfig) -> str:
    if config.save_prefix:
        return config.save_prefix

    if config.run_name:
        name = config.run_name
    else:
        safe_w = str(config.darcy_weight).replace(".", "p")
        name = f"{config.dataset_mode}_{config.training_mode}_{config.model_type}_darcy_{safe_w}"

    return os.path.join(config.save_dir, name)


def ensure_parent_dir(path_prefix: str):
    folder = os.path.dirname(path_prefix)
    if folder:
        os.makedirs(folder, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(path: str, history: Dict[str, Any]):
    curve_keys = [
        "train_loss_used",
        "train_total_mse",
        "train_supervised_mse",
        "train_mask_mse",
        "train_nonmask_mse",
        "train_darcy",
        "val_total_mse",
        "val_supervised_mse",
        "val_mask_mse",
        "val_nonmask_mse",
        "val_darcy",
    ]

    n_epochs = len(history["train_loss_used"])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + curve_keys)

        for i in range(n_epochs):
            row = [i + 1]
            for key in curve_keys:
                values = history.get(key, [])
                row.append(values[i] if i < len(values) else "")
            writer.writerow(row)


def make_run_summary(
    history: Dict[str, Any],
    config: ExperimentConfig,
    best_epoch: int,
    best_val_score: float,
) -> Dict[str, Any]:

    def best(key):
        values = history[key]
        return min(values) if len(values) else None

    def final(key):
        values = history[key]
        return values[-1] if len(values) else None

    return {
        "run_name": config.run_name,
        "model_type": config.model_type,
        "dataset_mode": config.dataset_mode,
        "training_mode": config.training_mode,
        "channels": config.channels,
        "mse_weight": config.mse_weight,
        "darcy_weight": config.darcy_weight,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "best_epoch": best_epoch,
        "best_val_score_used": best_val_score,
        "best_train_loss_used": best("train_loss_used"),
        "best_val_total_mse": best("val_total_mse"),
        "best_val_supervised_mse": best("val_supervised_mse"),
        "best_val_mask_mse": best("val_mask_mse"),
        "best_val_nonmask_mse": best("val_nonmask_mse"),
        "best_val_darcy": best("val_darcy"),
        "final_train_loss_used": final("train_loss_used"),
        "final_val_total_mse": final("val_total_mse"),
        "final_val_supervised_mse": final("val_supervised_mse"),
        "final_val_mask_mse": final("val_mask_mse"),
        "final_val_nonmask_mse": final("val_nonmask_mse"),
        "final_val_darcy": final("val_darcy"),
        "config": asdict(config),
    }


def save_run_outputs(
    path_prefix: str,
    model: nn.Module,
    history: Dict[str, Any],
    config: ExperimentConfig,
    best_epoch: int,
    best_val_score: float,
):
    """
    Saves:
        *_history.csv
        *_history.pt
        *_summary.json
        *_final_state.pt
    """
    ensure_parent_dir(path_prefix)

    save_history_csv(f"{path_prefix}_history.csv", history)
    torch.save(history, f"{path_prefix}_history.pt")

    summary = make_run_summary(history, config, best_epoch, best_val_score)
    save_json(f"{path_prefix}_summary.json", summary)

    torch.save(model.state_dict(), f"{path_prefix}_final_state.pt")


# -----------------------------------------------------------------------------
# Loss and metric helpers
# -----------------------------------------------------------------------------

def align_channels(label: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if label.shape[1] == out.shape[1]:
        return label
    if label.shape[1] > out.shape[1]:
        return label[:, : out.shape[1]]

    raise ValueError(f"Label has {label.shape[1]} channels but output has {out.shape[1]} channels.")


def expand_mask_for_channels(mask: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    return mask.expand(-1, out.shape[1], -1, -1)


def mse_on_region(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    diff2 = ((pred - target) ** 2) * mask
    denom = mask.sum()

    if denom.item() == 0:
        return torch.tensor(0.0, device=pred.device)

    return diff2.sum() / denom


def supervised_loss(
    out: torch.Tensor,
    label: torch.Tensor,
    mask: Optional[torch.Tensor],
    training_mode: TrainingMode,
    crit: nn.Module,
) -> torch.Tensor:

    label = align_channels(label, out)

    if training_mode == "baseline_full":
        return crit(out, label)

    if training_mode == "physics_limited":
        if mask is None:
            raise ValueError("physics_limited mode requires a mask.")

        mask_c = expand_mask_for_channels(mask.bool(), out).float()
        return mse_on_region(out, label, mask_c)

    raise ValueError(f"Unknown training_mode: {training_mode}")


def unpack_batch(batch, training_mode: TrainingMode):
    if training_mode == "physics_limited":
        feat, label, mask = batch
        return feat, label, mask

    if training_mode == "baseline_full":
        feat, label = batch
        return feat, label, None

    raise ValueError(f"Unknown training_mode: {training_mode}")


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    crit: nn.Module,
) -> Dict[str, float]:

    model.eval()

    total_mse = 0.0
    supervised_mse = 0.0
    mask_mse = 0.0
    nonmask_mse = 0.0
    darcy = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            feat, label, mask = unpack_batch(batch, config.training_mode)

            feat = feat.to(DEVICE)
            label = label.to(DEVICE)
            mask = mask.to(DEVICE).bool() if mask is not None else None

            out = model(feat)
            label = align_channels(label, out)

            total_mse += crit(out, label).item()
            supervised_mse += supervised_loss(out, label, mask, config.training_mode, crit).item()
            darcy += darcy_loss_from_output(out).item()

            if mask is not None:
                mask_c = expand_mask_for_channels(mask, out)
                nonmask_c = ~mask_c

                mask_mse += mse_on_region(out, label, mask_c).item()
                nonmask_mse += mse_on_region(out, label, nonmask_c).item()
            else:
                mask_mse += float("nan")
                nonmask_mse += float("nan")

            n_batches += 1

    return {
        "total_mse": total_mse / n_batches,
        "supervised_mse": supervised_mse / n_batches,
        "mask_mse": mask_mse / n_batches,
        "nonmask_mse": nonmask_mse / n_batches,
        "darcy": darcy / n_batches,
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def run_experiment(**kwargs):
    config = ExperimentConfig(**kwargs)

    if config.training_mode == "baseline_full" and config.darcy_weight != 0:
        raise ValueError("baseline_full is the no-physics baseline. Set darcy_weight=0.0.")

    path_prefix = make_run_prefix(config)
    ensure_parent_dir(path_prefix)

    model = make_model(config.model_type, config.channels)
    optimizer = Adam(model.parameters(), lr=config.lr)
    crit = nn.MSELoss()

    train_loader, val_loader = make_loaders(config)

    history = {
        "train_loss_used": [],
        "train_total_mse": [],
        "train_supervised_mse": [],
        "train_mask_mse": [],
        "train_nonmask_mse": [],
        "train_darcy": [],
        "val_total_mse": [],
        "val_supervised_mse": [],
        "val_mask_mse": [],
        "val_nonmask_mse": [],
        "val_darcy": [],
        "config": asdict(config),
    }

    best_val_score = float("inf")
    best_epoch = 0

    for epoch in tqdm(range(1, config.epochs + 1)):
        model.train()

        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            feat, label, mask = unpack_batch(batch, config.training_mode)

            feat = feat.to(DEVICE)
            label = label.to(DEVICE)
            mask = mask.to(DEVICE).bool() if mask is not None else None

            optimizer.zero_grad()

            out = model(feat)
            label = align_channels(label, out)

            mse = supervised_loss(out, label, mask, config.training_mode, crit)
            physics = darcy_loss_from_output(out)

            loss = config.mse_weight * mse + config.darcy_weight * physics

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss_used"].append(epoch_loss / n_batches)

        train_metrics = evaluate_loader(model, train_loader, config, crit)
        val_metrics = evaluate_loader(model, val_loader, config, crit)

        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            history[f"{split}_total_mse"].append(metrics["total_mse"])
            history[f"{split}_supervised_mse"].append(metrics["supervised_mse"])
            history[f"{split}_mask_mse"].append(metrics["mask_mse"])
            history[f"{split}_nonmask_mse"].append(metrics["nonmask_mse"])
            history[f"{split}_darcy"].append(metrics["darcy"])

        if config.best_metric is not None:
            val_score = val_metrics[config.best_metric]
        elif config.training_mode == "physics_limited":
            val_score = val_metrics["supervised_mse"]
        else:
            val_score = val_metrics["total_mse"]

        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch

            if config.save_best:
                torch.save(model.state_dict(), f"{path_prefix}_best_state.pt")

    metric_name = config.best_metric or (
        "val_supervised_mse" if config.training_mode == "physics_limited" else "val_total_mse"
    )

    print(f"Best epoch: {best_epoch}, best {metric_name}: {best_val_score:.6f}")

    save_run_outputs(path_prefix, model, history, config, best_epoch, best_val_score)

    return model, history


def run_experiment_grid(
    model_types=("splitnet_attn",),
    dataset_modes=("fixed",),
    training_modes=("physics_limited",),
    darcy_weights=(1.0,),
    epochs=50,
    base_save_dir="results",
    **kwargs,
):
    results = {}

    for dataset_mode in dataset_modes:
        for model_type in model_types:
            for training_mode in training_modes:

                weights = [0.0] if training_mode == "baseline_full" else darcy_weights

                for darcy_weight in weights:
                    safe_w = str(darcy_weight).replace(".", "p")

                    if training_mode == "baseline_full":
                        run_name = f"{dataset_mode}_{model_type}_baseline_full_nodarcy"
                    else:
                        run_name = f"{dataset_mode}_{model_type}_physics_limited_darcy_{safe_w}"

                    print(f"\n===== Running {run_name} =====")

                    model, history = run_experiment(
                        model_type=model_type,
                        dataset_mode=dataset_mode,
                        training_mode=training_mode,
                        darcy_weight=darcy_weight,
                        epochs=epochs,
                        save_dir=base_save_dir,
                        run_name=run_name,
                        **kwargs,
                    )

                    results[run_name] = {"model": model, "history": history}

    return results