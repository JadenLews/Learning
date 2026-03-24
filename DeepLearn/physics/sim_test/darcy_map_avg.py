import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

OUTPUT_DIR = "./output_quan"

# choose one: "jpg", "mat", "mat_u8"
SOURCE = "mat"

# choose simulation number
SIM_NO = 26


def load_gray(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read file: {path}")
    return arr.astype(np.float32)


def darcy_map_from_fields(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    dP_dy, dP_dx = np.gradient(P)

    y_flux = K * dP_dy
    x_flux = K * dP_dx

    d_yflux_dy = np.gradient(y_flux, axis=0)
    d_xflux_dx = np.gradient(x_flux, axis=1)

    residual = d_yflux_dy + d_xflux_dx
    return residual ** 2


def get_available_steps(output_dir: str, sim_no: int, source: str) -> list[int]:
    steps = []

    if source == "jpg":
        pattern = re.compile(rf"^Image-{sim_no}-(\d+)_P\.jpg$")
    else:
        pattern = re.compile(rf"^State-{sim_no}-(\d+)\.mat$")

    for fname in os.listdir(output_dir):
        match = pattern.match(fname)
        if match:
            steps.append(int(match.group(1)))

    return sorted(steps)


def load_fields(output_dir: str, sim_no: int, step: int, source: str):
    if source == "jpg":
        k_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_K.jpg")
        p_path = os.path.join(output_dir, f"Image-{sim_no}-{step}_P.jpg")

        K = load_gray(k_path) #/ 255.0
        P = load_gray(p_path) #/ 255.0
        return K, P

    mat_path = os.path.join(output_dir, f"State-{sim_no}-{step}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Could not find file: {mat_path}")

    data = loadmat(mat_path)

    if source == "mat":
        K = np.array(data["K_255"], dtype=np.float32).squeeze()
        P = np.array(data["P_255"], dtype=np.float32).squeeze()

    elif source == "mat_u8":
        K = np.array(data["K_u8"], dtype=np.float32).squeeze()
        P = np.array(data["P_u8"], dtype=np.float32).squeeze()
        K = K / 255.0
        P = P / 255.0

    else:
        raise ValueError("SOURCE must be one of: 'jpg', 'mat', 'mat_u8'")

    return K, P


def summarize_array(name: str, arr: np.ndarray):
    print(
        f"{name:18s} | shape={arr.shape} "
        f"min={arr.min():.6g} max={arr.max():.6g} "
        f"mean={arr.mean():.6g} std={arr.std():.6g}"
    )


def main():
    steps = get_available_steps(OUTPUT_DIR, SIM_NO, SOURCE)

    if not steps:
        raise FileNotFoundError(
            f"No steps found for sim {SIM_NO} in {OUTPUT_DIR} using source '{SOURCE}'"
        )

    print(f"Found {len(steps)} steps for sim {SIM_NO}: {steps[:10]}{' ...' if len(steps) > 10 else ''}")

    dmaps = []

    TOP_TRIM = 18
    BOTTOM_TRIM = 10
    for step in steps:
        try:
            K, P = load_fields(OUTPUT_DIR, SIM_NO, step, SOURCE)
            dmap = darcy_map_from_fields(K, P)

            # trim top and bottom
            dmap = dmap[TOP_TRIM:-BOTTOM_TRIM, :]

            dmaps.append(dmap)
        except Exception as e:
            print(f"Skipping step {step}: {e}")

    if not dmaps:
        raise RuntimeError("No Darcy maps were successfully computed.")

    dmaps = np.stack(dmaps, axis=0)
    avg_dmap = np.mean(dmaps, axis=0)

    print()
    summarize_array("Average Darcy", avg_dmap)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(avg_dmap, cmap="viridis")
    plt.title(f"Average Darcy Map\nsim={SIM_NO}, source={SOURCE}, n_steps={dmaps.shape[0]}")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()