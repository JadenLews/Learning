import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

OUTPUT_DIR = "../../Data200x200_withInfo"
OUTPUT_DIR = "./output_quan"


SIM_NO = 26
STEP = 150

# two sources
SOURCE1 = "jpg_255"
SOURCE2 = "mat_u8"

TOP_TRIM = 25
BOTTOM_TRIM = 15


def load_gray(path):
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read file: {path}")
    return arr.astype(np.float32)


def load_from_jpg(sim_no, step, output_dir = OUTPUT_DIR):
    k_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_K.jpg")
    p_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_P.jpg")
    phi_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_phi.jpg")

    K = load_gray(k_path)
    P = load_gray(p_path)
    phi = load_gray(phi_path)
    return K, P, phi


def load_from_mat(sim_no, step, output_dir = OUTPUT_DIR, variant = "raw"):
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
        raise ValueError("invalid")


def compute_darcy_parts(K: np.ndarray, P: np.ndarray):
    dP_dy, dP_dx = np.gradient(P)

    y_veloc = K * dP_dy
    x_veloc = K * dP_dx

    d_yveloc_dy = np.gradient(y_veloc, axis=0)
    d_xveloc_dx = np.gradient(x_veloc, axis=1)

    residual = d_yveloc_dy + d_xveloc_dx
    residual_sq = residual ** 2

    return {
        "P": P,
        "K": K,
        "dP_dy": dP_dy,
        "dP_dx": dP_dx,
        "y_veloc": y_veloc,
        "x_veloc": x_veloc,
        "d_yveloc_dy": d_yveloc_dy,
        "d_xveloc_dx": d_xveloc_dx,
        "residual": residual,
        "residual_sq": residual_sq,
    }


def trim_map(arr):
    return arr[TOP_TRIM:-BOTTOM_TRIM, :]


def summarize(name, arr):
    print(
        f"{name:16s} | shape={arr.shape} "
        f"min={arr.min():.6g} max={arr.max():.6g} "
        f"mean={arr.mean():.6g} std={arr.std():.6g}"
    )


def plot_part_comparison(parts1, parts2, label1, label2):
    keys = [
        "P", "K",
        "dP_dy", "dP_dx",
        "y_veloc", "x_veloc",
        "d_yveloc_dy", "d_xveloc_dx",
        "residual", "residual_sq"
    ]

    ncols = 3
    nrows = int(np.ceil(len(keys) / 2))

    fig, axs = plt.subplots(nrows, 4, figsize=(18, 2 * nrows))
    axs = np.array(axs).reshape(nrows, 4)

    for i, key in enumerate(keys):
        r = i // 2
        c_offset = (i % 2) * 2

        arr1 = trim_map(parts1[key])
        arr2 = trim_map(parts2[key])

        ax1 = axs[r, c_offset]
        ax2 = axs[r, c_offset + 1]

        im1 = ax1.imshow(arr1, cmap="viridis")
        ax1.set_title(f"{key}\n({label1})", fontsize=8)
        plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.01)

        im2 = ax2.imshow(arr2, cmap="viridis")
        ax2.set_title(f"{key}\n({label2})", fontsize=8)
        plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.01)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

    total_slots = nrows * 2
    if len(keys) < total_slots:
        for j in range(len(keys), total_slots):
            r = j // 2
            c_offset = (j % 2) * 2
            axs[r, c_offset].axis("off")
            axs[r, c_offset + 1].axis("off")

    plt.tight_layout(pad=0.5)
    plt.show()


def plot_difference_maps(parts1: dict, parts2: dict, label1: str, label2: str):
    keys = [
        "P", "K",
        "dP_dy", "dP_dx",
        "y_veloc", "x_veloc",
        "d_yveloc_dy", "d_xveloc_dx",
        "residual", "residual_sq"
    ]

    ncols = 4
    nrows = int(np.ceil(len(keys) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows))
    axs = np.array(axs).reshape(nrows, ncols)

    for i, key in enumerate(keys):
        r = i // ncols
        c = i % ncols
        ax = axs[r, c]

        arr1 = trim_map(parts1[key])
        arr2 = trim_map(parts2[key])

        # normalized comparison so different raw scales are easier to inspect
        a1 = (arr1 - arr1.min()) / (arr1.max() - arr1.min() + 1e-12)
        a2 = (arr2 - arr2.min()) / (arr2.max() - arr2.min() + 1e-12)
        diff = np.abs(a1 - a2)

        im = ax.imshow(diff, cmap="magma")
        ax.set_title(f"|normalized diff| of {key}\n{label1} vs {label2}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    # hide any unused axes
    for j in range(len(keys), nrows * ncols):
        r = j // ncols
        c = j % ncols
        axs[r, c].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    K1, P1, phi1 = load_source(SIM_NO, STEP, SOURCE1)
    K2, P2, phi2 = load_source(SIM_NO, STEP, SOURCE2)

    parts1 = compute_darcy_parts(K1, P1)
    parts2 = compute_darcy_parts(K2, P2)

    print(f"Comparing {SOURCE1} vs {SOURCE2}\n")

    for key in ["P", "K", "dP_dy", "dP_dx", "y_veloc", "x_veloc", "d_yveloc_dy", "d_xveloc_dx", "residual", "residual_sq"]:
        summarize(f"{key} ({SOURCE1})", trim_map(parts1[key]))
        summarize(f"{key} ({SOURCE2})", trim_map(parts2[key]))
        print()

    plot_part_comparison(parts1, parts2, SOURCE1, SOURCE2)
    plot_difference_maps(parts1, parts2, SOURCE1, SOURCE2)


if __name__ == "__main__":
    main()