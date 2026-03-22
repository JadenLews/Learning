import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

OUTPUT_DIR = "./output_folder"


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


def load_from_mat(sim_no: int, step: int, output_dir: str = OUTPUT_DIR, use_u8: bool = False):
    mat_path = os.path.join(output_dir, f"State-{sim_no}-{step}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Could not find file: {mat_path}")

    data = loadmat(mat_path)

    if use_u8:
        phi = np.array(data["phi_u8"], dtype=np.float32).squeeze()
        K   = np.array(data["K_u8"], dtype=np.float32).squeeze()
        P   = np.array(data["P_u8"], dtype=np.float32).squeeze()
    else:
        phi = np.array(data["phi"], dtype=np.float32).squeeze()
        K   = np.array(data["K"], dtype=np.float32).squeeze()
        P   = np.array(data["P"], dtype=np.float32).squeeze()

    VX = np.array(data["VX"], dtype=np.float32).squeeze()
    VY = np.array(data["VY"], dtype=np.float32).squeeze()

    return K, P, phi, VX, VY



def darcy_map_from_fields(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    dP_dy, dP_dx = np.gradient(P)

    y_flux = K * dP_dy
    x_flux = K * dP_dx

    d_yflux_dy = np.gradient(y_flux, axis=0)
    d_xflux_dx = np.gradient(x_flux, axis=1)

    residual = d_yflux_dy + d_xflux_dx
    return residual ** 2


def summarize_array(name: str, arr: np.ndarray):
    print(
        f"{name:12s} | shape={arr.shape} "
        f"min={arr.min():.6g} max={arr.max():.6g} "
        f"mean={arr.mean():.6g} std={arr.std():.6g}"
    )


def compare_fields(arr1: np.ndarray, arr2: np.ndarray, name: str, label1: str, label2: str):
    print(f"\n{name} comparison")
    summarize_array(f"{name} ({label1})", arr1)
    summarize_array(f"{name} ({label2})", arr2)

    arr1_norm = (arr1 - arr1.min()) / (arr1.max() - arr1.min() + 1e-12)
    arr2_norm = (arr2 - arr2.min()) / (arr2.max() - arr2.min() + 1e-12)

    diff = np.abs(arr1_norm - arr2_norm)
    print(
        f"{name} normalized abs diff | "
        f"mean={diff.mean():.6g} max={diff.max():.6g}"
    )


def plot_comparison(
    K1, P1, phi1, dmap1,
    K2, P2, phi2, dmap2,
    sim_no, step,
    label1="A", label2="B"
):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    # top row
    im = axs[0, 0].imshow(phi1, cmap="gray")
    axs[0, 0].set_title(f"phi ({label1})")
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im = axs[0, 1].imshow(P1, cmap="gray")
    axs[0, 1].set_title(f"P ({label1})")
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im = axs[0, 2].imshow(K1, cmap="gray")
    axs[0, 2].set_title(f"K ({label1})")
    plt.colorbar(im, ax=axs[0, 2], fraction=0.046, pad=0.04)

    im = axs[0, 3].imshow(dmap1, cmap="viridis")
    axs[0, 3].set_title(f"Darcy map ({label1})")
    plt.colorbar(im, ax=axs[0, 3], fraction=0.046, pad=0.04)

    # bottom row
    im = axs[1, 0].imshow(phi2, cmap="gray")
    axs[1, 0].set_title(f"phi ({label2})")
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im = axs[1, 1].imshow(P2, cmap="gray")
    axs[1, 1].set_title(f"P ({label2})")
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im = axs[1, 2].imshow(K2, cmap="gray")
    axs[1, 2].set_title(f"K ({label2})")
    plt.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)

    im = axs[1, 3].imshow(dmap2, cmap="viridis")
    axs[1, 3].set_title(f"Darcy map ({label2})")
    plt.colorbar(im, ax=axs[1, 3], fraction=0.046, pad=0.04)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Sim {sim_no}, Step {step}: {label1} vs {label2}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_residuals_side_by_side(dmap_mat: np.ndarray, dmap_jpg: np.ndarray):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im = axs[0].imshow(dmap_mat, cmap="viridis")
    axs[0].set_title("Darcy map from .mat")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    im = axs[1].imshow(dmap_jpg, cmap="viridis")
    axs[1].set_title("Darcy map from .jpg")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    sim_no = 26
    step = 8

    # choose one:
    # mode = "raw_vs_jpg"
    # mode = "u8_vs_jpg"
    mode = "raw_vs_jpg"

    if mode == "raw_vs_jpg":
        K1, P1, phi1, _, _ = load_from_mat(sim_no, step, use_u8=False)
        K2, P2, phi2 = load_from_jpg(sim_no, step)

        # jpg images need to be scaled back to [0,1]
        K2 = K2 / 255.0
        P2 = P2 / 255.0
        phi2 = phi2 / 255.0

        label1 = "mat_raw"
        label2 = "jpg"

    elif mode == "u8_vs_jpg":
        K1, P1, phi1, _, _ = load_from_mat(sim_no, step, use_u8=True)
        K2, P2, phi2 = load_from_jpg(sim_no, step)

        # both are on 0..255 scale now
        label1 = "mat_u8"
        label2 = "jpg"

    elif mode == "raw_vs_u8":
        K1, P1, phi1, _, _ = load_from_mat(sim_no, step, use_u8=False)
        K2, P2, phi2, _, _ = load_from_mat(sim_no, step, use_u8=True)

        # bring u8 back to [0,1] scale for fair comparison to raw
        K2 = K2 / 255.0
        P2 = P2 / 255.0
        phi2 = phi2 / 255.0

        label1 = "mat_raw"
        label2 = "mat_u8"

    else:
        raise ValueError("mode must be 'raw_vs_jpg', 'u8_vs_jpg', or 'raw_vs_u8'")

    dmap1 = darcy_map_from_fields(K1, P1)
    dmap2 = darcy_map_from_fields(K2, P2)

    # trim top/bottom rows
    top_trim = 25
    bottom_trim = 15

    dmap1 = dmap1[top_trim:-bottom_trim, :]
    dmap2 = dmap2[top_trim:-bottom_trim, :]

    print("\n=== RAW FIELD SUMMARY ===")
    compare_fields(phi1, phi2, "phi", label1, label2)
    compare_fields(P1, P2, "P", label1, label2)
    compare_fields(K1, K2, "K", label1, label2)

    print("\n=== DARCY MAP SUMMARY ===")
    summarize_array(f"Darcy ({label1})", dmap1)
    summarize_array(f"Darcy ({label2})", dmap2)

    ratio_mean = dmap2.mean() / (dmap1.mean() + 1e-20)
    ratio_max = dmap2.max() / (dmap1.max() + 1e-20)
    print(f"\nDarcy mean ratio ({label2} / {label1}): {ratio_mean:.6g}")
    print(f"Darcy max ratio  ({label2} / {label1}): {ratio_max:.6g}")

    plot_comparison(
        K1, P1, phi1, dmap1,
        K2, P2, phi2, dmap2,
        sim_no, step,
        label1=label1, label2=label2
    )

    plot_residuals_side_by_side(dmap1, dmap2)


if __name__ == "__main__":
    main()