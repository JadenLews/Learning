import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

OUTPUT_DIR = "./output_quan"

SIM_NO = 26
STEP = 200

# choose any two sources
SOURCE1 = "mat_u8"
SOURCE2 = "jpg_255"

TOP_TRIM = 25
BOTTOM_TRIM = 15


def load_gray(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read file: {path}")
    return arr.astype(np.float32)


def load_from_jpg(sim_no: int, step: int, output_dir: str = OUTPUT_DIR):
    k_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_K.jpg")
    p_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_P.jpg")
    phi_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_phi.jpg")

    K = load_gray(k_path)
    P = load_gray(p_path)
    phi = load_gray(phi_path)
    return K, P, phi


def load_from_mat(sim_no: int, step: int, output_dir: str = OUTPUT_DIR, variant: str = "raw"):
    mat_path = os.path.join(output_dir, f"State-{sim_no}-{step}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Could not find file: {mat_path}")

    data = loadmat(mat_path)

    if variant == "raw":
        phi = np.array(data["phi"], dtype=np.float32).squeeze()
        K   = np.array(data["K"], dtype=np.float32).squeeze()
        P   = np.array(data["P"], dtype=np.float32).squeeze()
    elif variant == "255":
        phi = np.array(data["phi_255"], dtype=np.float32).squeeze()
        K   = np.array(data["K_255"], dtype=np.float32).squeeze()
        P   = np.array(data["P_255"], dtype=np.float32).squeeze()
    elif variant == "u8":
        phi = np.array(data["phi_u8"], dtype=np.float32).squeeze()
        K   = np.array(data["K_u8"], dtype=np.float32).squeeze()
        P   = np.array(data["P_u8"], dtype=np.float32).squeeze()
    else:
        raise ValueError("variant must be 'raw', '255', or 'u8'")

    return K, P, phi


def load_source(sim_no: int, step: int, source: str):
    """
    source options:
        - mat_raw
        - mat_255
        - mat_u8
        - jpg_01
        - jpg_255
    """
    if source == "mat_raw":
        return load_from_mat(sim_no, step, variant="raw")
    elif source == "mat_255":
        return load_from_mat(sim_no, step, variant="255")
    elif source == "mat_u8":
        return load_from_mat(sim_no, step, variant="u8")
    elif source == "jpg_01":
        K, P, phi = load_from_jpg(sim_no, step)
        return K / 255.0, P / 255.0, phi / 255.0
    elif source == "jpg_255":
        return load_from_jpg(sim_no, step)
    else:
        raise ValueError("invalid source")


def compute_darcy_parts(K: np.ndarray, P: np.ndarray):
    dP_dy, dP_dx = np.gradient(P)

    y_flux = K * dP_dy
    x_flux = K * dP_dx

    d_yflux_dy = np.gradient(y_flux, axis=0)
    d_xflux_dx = np.gradient(x_flux, axis=1)

    residual = d_yflux_dy + d_xflux_dx
    residual_sq = residual ** 2

    return {
        "P": P,
        "K": K,
        "dP_dy": dP_dy,
        "dP_dx": dP_dx,
        "y_flux": y_flux,
        "x_flux": x_flux,
        "d_yflux_dy": d_yflux_dy,
        "d_xflux_dx": d_xflux_dx,
        "residual": residual,
        "residual_sq": residual_sq,
    }


def trim_map(arr: np.ndarray) -> np.ndarray:
    return arr[TOP_TRIM:-BOTTOM_TRIM, :]


def summarize(name: str, arr: np.ndarray):
    print(
        f"{name:16s} | shape={arr.shape} "
        f"min={arr.min():.6g} max={arr.max():.6g} "
        f"mean={arr.mean():.6g} std={arr.std():.6g}"
    )


def plot_part_comparison(parts1: dict, parts2: dict, label1: str, label2: str):
    keys = [
        "P", "K",
        "dP_dy", "dP_dx",
        "y_flux", "x_flux",
        "d_yflux_dy", "d_xflux_dx",
        "residual", "residual_sq"
    ]

    fig, axs = plt.subplots(len(keys), 2, figsize=(10, 3 * len(keys)))

    for i, key in enumerate(keys):
        arr1 = trim_map(parts1[key])
        arr2 = trim_map(parts2[key])

        im1 = axs[i, 0].imshow(arr1, cmap="viridis")
        axs[i, 0].set_title(f"{key} ({label1})")
        plt.colorbar(im1, ax=axs[i, 0], fraction=0.046, pad=0.04)

        im2 = axs[i, 1].imshow(arr2, cmap="viridis")
        axs[i, 1].set_title(f"{key} ({label2})")
        plt.colorbar(im2, ax=axs[i, 1], fraction=0.046, pad=0.04)

        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_difference_maps(parts1: dict, parts2: dict, label1: str, label2: str):
    keys = [
        "P", "K",
        "dP_dy", "dP_dx",
        "y_flux", "x_flux",
        "d_yflux_dy", "d_xflux_dx",
        "residual", "residual_sq"
    ]

    fig, axs = plt.subplots(len(keys), 1, figsize=(6, 3 * len(keys)))

    for i, key in enumerate(keys):
        arr1 = trim_map(parts1[key])
        arr2 = trim_map(parts2[key])

        # normalized comparison so different raw scales are easier to inspect
        a1 = (arr1 - arr1.min()) / (arr1.max() - arr1.min() + 1e-12)
        a2 = (arr2 - arr2.min()) / (arr2.max() - arr2.min() + 1e-12)
        diff = np.abs(a1 - a2)

        im = axs[i].imshow(diff, cmap="magma")
        axs[i].set_title(f"|normalized diff| of {key}: {label1} vs {label2}")
        plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    K1, P1, phi1 = load_source(SIM_NO, STEP, SOURCE1)
    K2, P2, phi2 = load_source(SIM_NO, STEP, SOURCE2)

    parts1 = compute_darcy_parts(K1, P1)
    parts2 = compute_darcy_parts(K2, P2)

    print(f"Comparing {SOURCE1} vs {SOURCE2}\n")

    for key in ["P", "K", "dP_dy", "dP_dx", "y_flux", "x_flux", "d_yflux_dy", "d_xflux_dx", "residual", "residual_sq"]:
        summarize(f"{key} ({SOURCE1})", trim_map(parts1[key]))
        summarize(f"{key} ({SOURCE2})", trim_map(parts2[key]))
        print()

    plot_part_comparison(parts1, parts2, SOURCE1, SOURCE2)
    plot_difference_maps(parts1, parts2, SOURCE1, SOURCE2)


if __name__ == "__main__":
    main()