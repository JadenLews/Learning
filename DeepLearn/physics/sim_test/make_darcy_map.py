import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "/Users/jaden/projects/Learning/DeepLearn/physics/sim_test/output_modified_press_low"
OUTPUT_DIR = "/Users/jaden/projects/Learning/DeepLearn/physics/sim_test/output2"



def load_gray(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read file: {path}")
    return arr.astype(np.float32)


def load_fields(sim_no: int, step: int, output_dir: str = OUTPUT_DIR):
    """
    Loads K and P images from the output directory.
    """
    k_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_K.jpg")
    p_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_P.jpg")
    phi_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_phi.jpg")

    K = load_gray(k_path)
    P = load_gray(p_path)
    phi = load_gray(phi_path)

    return K, P, phi


def darcy_map_from_fields(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Computes a Darcy residual map similar to:
        div(K * grad(P))^2

    Uses numpy.gradient, which is a reasonable approximation
    to what your PyTorch code is doing.
    """
    # grad(P): [dP/dy, dP/dx]
    dP_dy, dP_dx = np.gradient(P)

    y_flux = K * dP_dy
    x_flux = K * dP_dx

    d_yflux_dy = np.gradient(y_flux, axis=0)
    d_xflux_dx = np.gradient(x_flux, axis=1)

    residual = d_yflux_dy + d_xflux_dx
    darcy_map = residual ** 2

    return darcy_map


def plot_darcy_map(darcy_map: np.ndarray, title: str = "Darcy Residual Map", zoom: bool = False):
    plt.figure(figsize=(6, 5))
    plt.imshow(darcy_map, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    if zoom:
        plt.xlim(0, darcy_map.shape[1] - 1)
        plt.ylim(darcy_map.shape[0] - 1, 0)
    plt.tight_layout()
    plt.show()


def plot_full_and_zoom(darcy_map: np.ndarray, zoom_rows=(10, -10), zoom_cols=(0, None)):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # full map
    im0 = axs[0].imshow(darcy_map, cmap="viridis")
    axs[0].set_title("Darcy Loss Map")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    r0, r1 = zoom_rows
    c0, c1 = zoom_cols

    if c1 is None:
        zoom_map = darcy_map[r0:r1, :]
    else:
        zoom_map = darcy_map[r0:r1, c0:c1]

    # zoomed map
    im1 = axs[1].imshow(zoom_map, cmap="viridis")
    axs[1].set_title("Interior (Top/Bottom Trimmed)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def main():
    sim_no = 26
    step = 70

    K, P, phi = load_fields(sim_no, step)
    dmap = darcy_map_from_fields(K, P)

    print(f"Darcy map shape: {dmap.shape}")
    print(f"Mean Darcy loss: {dmap.mean():.6g}")
    print(f"Max Darcy loss:  {dmap.max():.6g}")

    plot_full_and_zoom(dmap)


if __name__ == "__main__":
    main()