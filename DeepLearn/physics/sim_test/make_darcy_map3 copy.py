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


def load_from_mat(sim_no: int, step: int, output_dir: str = OUTPUT_DIR):
    mat_path = os.path.join(output_dir, f"State-{sim_no}-{step}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Could not find file: {mat_path}")

    data = loadmat(mat_path)
    phi = np.array(data["phi"], dtype=np.float32).squeeze()
    K   = np.array(data["K"], dtype=np.float32).squeeze()
    P   = np.array(data["P"], dtype=np.float32).squeeze()
    VX  = np.array(data["VX"], dtype=np.float32).squeeze()
    VY  = np.array(data["VY"], dtype=np.float32).squeeze()
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


def compare_fields(mat_arr: np.ndarray, jpg_arr: np.ndarray, name: str):
    print(f"\n{name} comparison")
    summarize_array(f"{name} (.mat)", mat_arr)
    summarize_array(f"{name} (.jpg)", jpg_arr)

    # compare normalized versions just to see shape similarity
    mat_norm = (mat_arr - mat_arr.min()) / (mat_arr.max() - mat_arr.min() + 1e-12)
    jpg_norm = (jpg_arr - jpg_arr.min()) / (jpg_arr.max() - jpg_arr.min() + 1e-12)

    diff = np.abs(mat_norm - jpg_norm)
    print(
        f"{name} normalized abs diff | "
        f"mean={diff.mean():.6g} max={diff.max():.6g}"
    )


def plot_comparison(
    K_mat, P_mat, phi_mat, dmap_mat,
    K_jpg, P_jpg, phi_jpg, dmap_jpg,
    sim_no, step
):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    # top row = MAT
    im = axs[0, 0].imshow(phi_mat, cmap="gray")
    axs[0, 0].set_title("phi (.mat)")
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im = axs[0, 1].imshow(P_mat, cmap="gray")
    axs[0, 1].set_title("P (.mat)")
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im = axs[0, 2].imshow(K_mat, cmap="gray")
    axs[0, 2].set_title("K (.mat)")
    plt.colorbar(im, ax=axs[0, 2], fraction=0.046, pad=0.04)

    im = axs[0, 3].imshow(dmap_mat, cmap="viridis")
    axs[0, 3].set_title("Darcy map (.mat)")
    plt.colorbar(im, ax=axs[0, 3], fraction=0.046, pad=0.04)

    # bottom row = JPG
    im = axs[1, 0].imshow(phi_jpg, cmap="gray")
    axs[1, 0].set_title("phi (.jpg)")
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im = axs[1, 1].imshow(P_jpg, cmap="gray")
    axs[1, 1].set_title("P (.jpg)")
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im = axs[1, 2].imshow(K_jpg, cmap="gray")
    axs[1, 2].set_title("K (.jpg)")
    plt.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)

    im = axs[1, 3].imshow(dmap_jpg, cmap="viridis")
    axs[1, 3].set_title("Darcy map (.jpg)")
    plt.colorbar(im, ax=axs[1, 3], fraction=0.046, pad=0.04)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Sim {sim_no}, Step {step}: .mat vs .jpg comparison", fontsize=14)
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
    step = 85

    K_mat, P_mat, phi_mat, VX_mat, VY_mat = load_from_mat(sim_no, step)
    K_jpg, P_jpg, phi_jpg = load_from_jpg(sim_no, step)

    P_jpg = P_jpg / 255.0
    K_jpg = K_jpg / 255.0
    phi_jpg = phi_jpg / 255.0


    dmap_mat = darcy_map_from_fields(K_mat, P_mat)
    dmap_jpg = darcy_map_from_fields(K_jpg, P_jpg)

    # trim top/bottom rows
    top_trim = 25
    bottom_trim = 15

    dmap_mat = dmap_mat[top_trim:-bottom_trim, :]
    dmap_jpg = dmap_jpg[top_trim:-bottom_trim, :]

    print("\n=== RAW FIELD SUMMARY ===")
    compare_fields(phi_mat, phi_jpg, "phi")
    compare_fields(P_mat, P_jpg, "P")
    compare_fields(K_mat, K_jpg, "K")

    print("\n=== DARCY MAP SUMMARY ===")
    summarize_array("Darcy (.mat)", dmap_mat)
    summarize_array("Darcy (.jpg)", dmap_jpg)

    ratio_mean = dmap_jpg.mean() / (dmap_mat.mean() + 1e-20)
    ratio_max = dmap_jpg.max() / (dmap_mat.max() + 1e-20)
    print(f"\nDarcy mean ratio (.jpg / .mat): {ratio_mean:.6g}")
    print(f"Darcy max ratio  (.jpg / .mat): {ratio_max:.6g}")

    plot_comparison(
        K_mat, P_mat, phi_mat, dmap_mat,
        K_jpg, P_jpg, phi_jpg, dmap_jpg,
        sim_no, step
    )

    plot_residuals_side_by_side(dmap_mat, dmap_jpg)


if __name__ == "__main__":
    main()