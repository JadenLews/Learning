import torch
import torch.nn.functional as F


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def align_channels(label, out):
    if label.shape[1] == out.shape[1]:
        return label
    if label.shape[1] > out.shape[1]:
        return label[:, :out.shape[1]]
    raise ValueError("Label has fewer channels than output.")


def expand_mask_for_channels(mask, tensor):
    """
    Converts mask to [B, C, H, W].
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if mask.shape[1] == 1 and tensor.shape[1] > 1:
        mask = mask.expand(-1, tensor.shape[1], -1, -1)

    return mask


def reduce_error_map(error_map, region="all", mask=None, target=None, channel=None, boundary_width=10, high_grad_quantile=0.90):
    """
    Reduces a per-pixel error map to a scalar.

    error_map:
        [B, C, H, W]

    region options:
        "all"
        "mask"
        "nonmask"
        "boundary"
        "interior"
        "high_gradient"

    mask:
        dataset mask, needed for "mask" and "nonmask"

    target:
        needed for "high_gradient"

    channel:
        optional channel to restrict metric to one channel
    """

    if channel is not None:
        error_map = error_map[:, channel:channel + 1]

    if region == "all":
        return error_map.mean()

    if region in ["mask", "nonmask"]:
        if mask is None:
            raise ValueError(f"region='{region}' requires mask.")

        region_mask = expand_mask_for_channels(mask.bool(), error_map)

        if region == "nonmask":
            region_mask = ~region_mask

    elif region in ["boundary", "interior"]:
        region_mask = boundary_mask_like(error_map, width=boundary_width)

        if region == "interior":
            region_mask = ~region_mask

    elif region == "high_gradient":
        if target is None:
            raise ValueError("region='high_gradient' requires target.")

        if channel is None:
            # Use all channels together
            region_mask = high_gradient_mask(target, quantile=high_grad_quantile)
        else:
            region_mask = high_gradient_mask(target, channel=channel, quantile=high_grad_quantile)

        region_mask = expand_mask_for_channels(region_mask.bool(), error_map)

    else:
        raise ValueError(f"Unknown region: {region}")

    region_mask = region_mask.float()

    denom = region_mask.sum()

    if denom.item() == 0:
        return torch.tensor(0.0, device=error_map.device)

    return (error_map * region_mask).sum() / denom


# --------------------------------------------------
# Error maps
# --------------------------------------------------

def mse_map(pred, target):
    return (pred - target) ** 2


def abs_error_map(pred, target):
    return torch.abs(pred - target)


def gradient_difference_map(pred, target):
    """
    Per-pixel gradient magnitude difference.
    """
    pred_grad = gradient_magnitude_map(pred)
    true_grad = gradient_magnitude_map(target)

    return (pred_grad - true_grad) ** 2


# --------------------------------------------------
# Main metric interface
# --------------------------------------------------

def compute_metric(
    name,
    pred,
    target=None,
    mask=None,
    region="all",
    channel=None,
    **kwargs,
):
    """
    General metric function.

    Examples:
        compute_metric("mse", pred, target, region="all")
        compute_metric("mse", pred, target, mask=mask, region="mask")
        compute_metric("mse", pred, target, mask=mask, region="nonmask")
        compute_metric("mse", pred, target, region="high_gradient", channel=0)
        compute_metric("ssim", pred, target)
        compute_metric("darcy", pred)
        compute_metric("darcy_match", pred, target)
    """

    if target is not None:
        target = align_channels(target, pred)

    if name == "mse":
        require_target(name, target)
        error = mse_map(pred, target)
        return reduce_error_map(
            error,
            region=region,
            mask=mask,
            target=target,
            channel=channel,
            **kwargs,
        )

    if name == "abs_error":
        error = abs_error_map(pred, target)
        return reduce_error_map(
            error,
            region=region,
            mask=mask,
            target=target,
            channel=channel,
            **kwargs,
        )

    if name == "gradient":
        error = gradient_difference_map(pred, target)
        return reduce_error_map(
            error,
            region=region,
            mask=mask,
            target=target,
            channel=channel,
            **kwargs,
        )

    if name == "darcy":
        residual = darcy_residual_map(pred)
        error = residual ** 2
        return reduce_error_map(
            error,
            region=region,
            mask=mask,
            target=target,
            channel=None,
            **kwargs,
        )

    if name == "darcy_match":
        pred_res = darcy_residual_map(pred)
        true_res = darcy_residual_map(target)
        error = (pred_res - true_res) ** 2
        return reduce_error_map(
            error,
            region=region,
            mask=mask,
            target=target,
            channel=None,
            **kwargs,
        )

    if name == "ssim":
        if region != "all":
            raise ValueError("SSIM should usually be used with region='all'.")
        return ssim(pred, target, **kwargs)

    if name == "ssim_loss":
        if region != "all":
            raise ValueError("SSIM loss should usually be used with region='all'.")
        return 1.0 - ssim(pred, target, **kwargs)

    if name == "psnr":
        if region != "all":
            raise ValueError("PSNR should usually be used with region='all'.")
        return psnr(pred, target, **kwargs)
    
    if name == "tv":
        return total_variation_loss(
            pred,
            channel=channel,
        )

    if name == "channel_stats":
        return channel_mean_std_prior(
            pred,
            channel_mean=kwargs["channel_mean"],
            channel_std=kwargs["channel_std"],
            channels=kwargs.get("channels", None),
        )

    if name == "low_k_area":
        return low_k_area_prior(
            pred,
            target_area=kwargs["target_area"],
            threshold=kwargs.get("threshold", -0.5),
            channel=kwargs.get("channel", 0),
            softness=kwargs.get("softness", 20.0),
        )

    raise ValueError(f"Unknown metric name: {name}")


def darcy_residual_map(out):
    if out.shape[1] < 2:
        raise ValueError("Darcy residual requires at least K and P channels.")

    k = out[:, 0:1]
    p = out[:, 1:2]

    p_y, p_x = torch.gradient(p, dim=(-2, -1))

    flux_y = k * p_y
    flux_x = k * p_x

    div_y = torch.gradient(flux_y, dim=(-2,))[0]
    div_x = torch.gradient(flux_x, dim=(-1,))[0]

    return div_y + div_x


def gradient_magnitude_map(x, channel=None):
    if channel is not None:
        x = x[:, channel:channel + 1]

    gy, gx = torch.gradient(x, dim=(-2, -1))
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def high_gradient_mask(target, channel=None, quantile=0.90):
    grad_mag = gradient_magnitude_map(target, channel=channel)

    if channel is None:
        grad_mag = grad_mag.mean(dim=1, keepdim=True)

    flat = grad_mag.flatten(start_dim=1)
    thresh = torch.quantile(flat, quantile, dim=1)
    thresh = thresh.view(-1, 1, 1, 1)

    return grad_mag >= thresh


def boundary_mask_like(tensor, width=10):
    B, _, H, W = tensor.shape

    mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=tensor.device)

    mask[:, :, :width, :] = True
    mask[:, :, -width:, :] = True
    mask[:, :, :, :width] = True
    mask[:, :, :, -width:] = True

    return mask


def psnr(pred, target, data_range=2.0):
    mse = ((pred - target) ** 2).mean()

    if mse.item() == 0:
        return torch.tensor(float("inf"), device=pred.device)

    return 20 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10 * torch.log10(mse)


def ssim(pred, target, data_range=2.0, window_size=11):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    padding = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x2
    sigma_y2 = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return (numerator / (denominator + 1e-8)).mean()


def require_target(name, target):
    if target is None:
        raise ValueError(f"Metric '{name}' requires a target.")
    
def total_variation_loss(pred, channel=None):
    """
    Smoothness prior.

    Does not use target.
    Penalizes noisy/spiky predictions.
    Useful especially for pressure P.
    """
    if channel is not None:
        pred = pred[:, channel:channel + 1]

    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()

    return dx + dy


def channel_mean_std_prior(pred, channel_mean, channel_std, channels=None):
    """
    Matches prediction-level channel mean/std to dataset-level priors.

    Does not use the individual target.
    channel_mean and channel_std should come from dataset_priors.json.
    """

    device = pred.device
    dtype = pred.dtype

    mean_prior = torch.tensor(channel_mean, device=device, dtype=dtype)
    std_prior = torch.tensor(channel_std, device=device, dtype=dtype)

    if channels is not None:
        pred = pred[:, channels]
        mean_prior = mean_prior[channels]
        std_prior = std_prior[channels]

    pred_mean = pred.mean(dim=(0, 2, 3))
    pred_std = pred.std(dim=(0, 2, 3))

    mean_loss = ((pred_mean - mean_prior) ** 2).mean()
    std_loss = ((pred_std - std_prior) ** 2).mean()

    return mean_loss + std_loss


def low_k_area_prior(pred, target_area, threshold=-0.5, channel=0, softness=20.0):
    """
    Soft area prior for low-K regions.

    Does not use target.
    Encourages the predicted K channel to contain a realistic amount
    of low-conductivity/channel pixels.

    Uses sigmoid instead of hard threshold so it is differentiable.
    """

    k = pred[:, channel:channel + 1]

    low_k_soft = torch.sigmoid(softness * (threshold - k))
    pred_area = low_k_soft.mean()

    target_area = torch.tensor(
        target_area,
        device=pred.device,
        dtype=pred.dtype,
    )

    return (pred_area - target_area) ** 2