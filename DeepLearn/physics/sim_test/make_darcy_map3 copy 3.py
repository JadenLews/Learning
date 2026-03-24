import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

OUTPUT_DIR = "./output_quan"


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


def load_from_mat(
    sim_no: int,
    step: int,
    output_dir: str = OUTPUT_DIR,
    variant: str = "raw"
):
    """
    variant options:
        - 'raw' : original phi, P, K
        - '255' : continuous 0-255 scaled versions (phi_255, P_255, K_255)
        - 'u8'  : quantized 0-255 versions (phi_u8, P_u8, K_u8)
    """
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

    VX = np.array(data["VX"], dtype=np.float32).squeeze()
    VY = np.array(data["VY"], dtype=np.float32).squeeze()

    return K, P, phi, VX, VY


def load_source(sim_no: int, step: int, source: str, output_dir: str = OUTPUT_DIR):
    """
    source options:
        - 'mat_raw' : original continuous simulation values
        - 'mat_255' : continuous values scaled to 0..255
        - 'mat_u8'  : quantized 0..255 values, no jpg compression
        - 'jpg_01'  : jpg loaded and divided by 255
        - 'jpg_255' : jpg loaded in 0..255 space
    """
    if source == "mat_raw":
        K, P, phi, _, _ = load_from_mat(sim_no, step, output_dir=output_dir, variant="raw")
        return K, P, phi

    elif source == "mat_255":
        K, P, phi, _, _ = load_from_mat(sim_no, step, output_dir=output_dir, variant="255")
        return K, P, phi

    elif source == "mat_u8":
        K, P, phi, _, _ = load_from_mat(sim_no, step, output_dir=output_dir, variant="u8")
        return K, P, phi

    elif source == "jpg_01":
        K, P, phi = load_from_jpg(sim_no, step, output_dir=output_dir)
        return K / 255.0, P / 255.0, phi / 255.0

    elif source == "jpg_255":
        K, P, phi = load_from_jpg(sim_no, step, output_dir=output_dir)
        return K, P, phi

    else:
        raise ValueError(
            "source must be one of: 'mat_raw', 'mat_255', 'mat_u8', 'jpg_01', 'jpg_255'"
        )


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
        f"{name:18s} | shape={arr.shape} "
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


def plot_residuals_side_by_side(dmap1: np.ndarray, dmap2: np.ndarray, label1: str, label2: str):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im = axs[0].imshow(dmap1, cmap="viridis")
    axs[0].set_title(f"Darcy map ({label1})")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    im = axs[1].imshow(dmap2, cmap="viridis")
    axs[1].set_title(f"Darcy map ({label2})")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    sim_no = 26
    step = 45

    # choose any two sources you want to compare
    source1 = "mat_raw"
    source2 = "jpg_255"

    # useful examples:
    # source1 = "mat_raw"; source2 = "mat_255"   # pure scaling effect
    # source1 = "mat_255"; source2 = "mat_u8"    # quantization effect
    # source1 = "mat_u8";  source2 = "jpg_255"   # jpg compression effect
    # source1 = "mat_raw"; source2 = "jpg_01"    # full raw vs jpg pipeline

    K1, P1, phi1 = load_source(sim_no, step, source1)
    K2, P2, phi2 = load_source(sim_no, step, source2)

    dmap1 = darcy_map_from_fields(K1, P1)
    dmap2 = darcy_map_from_fields(K2, P2)

    top_trim = 25
    bottom_trim = 15

    dmap1 = dmap1[top_trim:-bottom_trim, :]
    dmap2 = dmap2[top_trim:-bottom_trim, :]

    print("OUTPUT_DIR =", OUTPUT_DIR)
    print(f"Comparing {source1} vs {source2}")

    print("\n=== FIELD SUMMARY ===")
    compare_fields(phi1, phi2, "phi", source1, source2)
    compare_fields(P1, P2, "P", source1, source2)
    compare_fields(K1, K2, "K", source1, source2)

    print("\n=== DARCY MAP SUMMARY ===")
    summarize_array(f"Darcy ({source1})", dmap1)
    summarize_array(f"Darcy ({source2})", dmap2)

    ratio_mean = dmap2.mean() / (dmap1.mean() + 1e-20)
    ratio_max = dmap2.max() / (dmap1.max() + 1e-20)
    print(f"\nDarcy mean ratio ({source2} / {source1}): {ratio_mean:.6g}")
    print(f"Darcy max ratio  ({source2} / {source1}): {ratio_max:.6g}")

    plot_comparison(
        K1, P1, phi1, dmap1,
        K2, P2, phi2, dmap2,
        sim_no, step,
        label1=source1, label2=source2
    )

    plot_residuals_side_by_side(dmap1, dmap2, source1, source2)


if __name__ == "__main__":
    main()