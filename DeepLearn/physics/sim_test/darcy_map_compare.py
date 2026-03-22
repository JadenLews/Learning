import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#directory where .mat AND .jpg values runs are
OUTPUT_DIR = os.path.join(THIS_DIR, "output_folder")


#choose sim and step to look at
sim_no = 26
step = 85


def load_gray(path):
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Can't read file: {path}")
    return arr.astype(np.float32)


# load files
mat_path = os.path.join(OUTPUT_DIR, f"State-{sim_no}-{step}.mat")
phi_path = os.path.join(OUTPUT_DIR, f"Image-{sim_no}-{step}_phi.jpg")
p_path   = os.path.join(OUTPUT_DIR, f"Image-{sim_no}-{step}_P.jpg")
k_path   = os.path.join(OUTPUT_DIR, f"Image-{sim_no}-{step}_K.jpg")

if not os.path.exists(mat_path):
    raise FileNotFoundError(f"Can't find file: {mat_path}")

data = loadmat(mat_path)

phi_mat = np.array(data["phi"], dtype=np.float32).squeeze()
P_mat   = np.array(data["P"], dtype=np.float32).squeeze()
K_mat   = np.array(data["K"], dtype=np.float32).squeeze()


# jpg values are 0-255, scale them down to 0-1 range like .mat values
phi_jpg = load_gray(phi_path) / 255.0
P_jpg   = load_gray(p_path) / 255.0
K_jpg   = load_gray(k_path) / 255.0


#  Darcy residual maps
dPdy_mat, dPdx_mat = np.gradient(P_mat)
yflux_mat = K_mat * dPdy_mat
xflux_mat = K_mat * dPdx_mat
dmap_mat = (np.gradient(yflux_mat, axis=0) + np.gradient(xflux_mat, axis=1)) ** 2

dPdy_jpg, dPdx_jpg = np.gradient(P_jpg)
yflux_jpg = K_jpg * dPdy_jpg
xflux_jpg = K_jpg * dPdx_jpg
dmap_jpg = (np.gradient(yflux_jpg, axis=0) + np.gradient(xflux_jpg, axis=1)) ** 2






# trim if top or bottom values are too extreme
top_trim = 25
bottom_trim = 15
dmap_mat_trim = dmap_mat[top_trim:-bottom_trim, :]
dmap_jpg_trim = dmap_jpg[top_trim:-bottom_trim, :]



# plots 
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

axs[0, 0].imshow(phi_mat, cmap="gray")
axs[0, 0].set_title("phi (.mat)")

axs[0, 1].imshow(P_mat, cmap="gray")
axs[0, 1].set_title("P (.mat)")

axs[0, 2].imshow(K_mat, cmap="gray")
axs[0, 2].set_title("K (.mat)")

axs[0, 3].imshow(dmap_mat_trim, cmap="viridis")
axs[0, 3].set_title("Darcy (.mat)")

axs[1, 0].imshow(phi_jpg, cmap="gray")
axs[1, 0].set_title("phi (.jpg)")

axs[1, 1].imshow(P_jpg, cmap="gray")
axs[1, 1].set_title("P (.jpg)")

axs[1, 2].imshow(K_jpg, cmap="gray")
axs[1, 2].set_title("K (.jpg)")

axs[1, 3].imshow(dmap_jpg_trim, cmap="viridis")
axs[1, 3].set_title("Darcy (.jpg)")

for ax in axs.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()