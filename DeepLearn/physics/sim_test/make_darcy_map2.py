import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

OUTPUT_DIR = "/Users/jaden/projects/Learning/DeepLearn/physics/sim_test/output_modified_press"
OUTPUT_DIR = "/Users/jaden/projects/Learning/DeepLearn/physics/sim_test/output1"


def load_fields_from_mat(sim_no: int, step: int, output_dir: str = OUTPUT_DIR):
    """
    Loads raw K, P, phi, VX, VY arrays from:
        State-{sim_no}-{step}.mat
    """
    mat_path = os.path.join(output_dir, f"State-{sim_no}-{step}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Could not find file: {mat_path}")

    data = loadmat(mat_path)

    # squeeze to remove extra singleton dims if present
    phi = np.array(data["phi"], dtype=np.float32).squeeze()
    K   = np.array(data["K"], dtype=np.float32).squeeze()
    P   = np.array(data["P"], dtype=np.float32).squeeze()
    VX  = np.array(data["VX"], dtype=np.float32).squeeze()
    VY  = np.array(data["VY"], dtype=np.float32).squeeze()

    return K, P, phi, VX, VY


def darcy_map_from_fields(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Computes Darcy residual map:
        div(K * grad(P))^2
    """
    dP_dy, dP_dx = np.gradient(P)

    y_flux = K * dP_dy
    x_flux = K * dP_dx

    d_yflux_dy = np.gradient(y_flux, axis=0)
    d_xflux_dx = np.gradient(x_flux, axis=1)

    residual = d_yflux_dy + d_xflux_dx
    darcy_map = residual ** 2

    return darcy_map


def divergence_map_from_velocity(VX: np.ndarray, VY: np.ndarray) -> np.ndarray:
    """
    Computes squared divergence map:
        (dVX/dx + dVY/dy)^2
    """
    dVX_dx = np.gradient(VX, axis=1)
    dVY_dy = np.gradient(VY, axis=0)

    div = dVX_dx + dVY_dy
    return div ** 2


def plot_full_and_zoom(arr: np.ndarray, title="Map", zoom_rows=(10, -10), zoom_cols=(0, None)):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(arr, cmap="viridis")
    axs[0].set_title(title)
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    r0, r1 = zoom_rows
    c0, c1 = zoom_cols

    if c1 is None:
        zoom_map = arr[r0:r1, :]
    else:
        zoom_map = arr[r0:r1, c0:c1]

    im1 = axs[1].imshow(zoom_map, cmap="viridis")
    axs[1].set_title(f"{title} (Interior)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def main():
    sim_no = 26
    step = 70

    K, P, phi, VX, VY = load_fields_from_mat(sim_no, step)

    dmap = darcy_map_from_fields(K, P)
    vmap = divergence_map_from_velocity(VX, VY)

    print(f"K shape: {K.shape}, P shape: {P.shape}, phi shape: {phi.shape}")
    print(f"Darcy map shape: {dmap.shape}")
    print(f"Mean Darcy loss: {dmap.mean():.6g}")
    print(f"Max Darcy loss:  {dmap.max():.6g}")

    plot_full_and_zoom(dmap, title="Darcy Loss Map")
    plot_full_and_zoom(vmap, title="Velocity Divergence Map")


if __name__ == "__main__":
    main()