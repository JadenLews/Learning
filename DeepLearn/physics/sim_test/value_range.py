import os
import re
import numpy as np
from scipy.io import loadmat

INPUT_DIR = "./output_quan"
SIM_NO = 26
STEP_INTERVAL = 40

FIELDS = ["phi", "P", "K", "VX", "VY"]


def get_steps(input_dir, sim_no):
    pattern = re.compile(rf"^State-{sim_no}-(\d+)\.mat$")
    steps = []
    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def field_stats(arr):
    return arr.min(), arr.max(), arr.mean(), arr.std()


def main():
    steps = get_steps(INPUT_DIR, SIM_NO)
    if not steps:
        raise FileNotFoundError(f"No State-{SIM_NO}-*.mat files found in {INPUT_DIR}")

    overall = {f: {"min": np.inf, "max": -np.inf} for f in FIELDS}

    print(f"\nSimulation {SIM_NO}")
    print(f"Found {len(steps)} steps: {steps[0]} to {steps[-1]}")

    print("\nEvery 40-step snapshot:")
    for step in steps:
        path = os.path.join(INPUT_DIR, f"State-{SIM_NO}-{step}.mat")
        data = loadmat(path)

        for f in FIELDS:
            arr = np.array(data[f], dtype=np.float32).squeeze()
            overall[f]["min"] = min(overall[f]["min"], float(arr.min()))
            overall[f]["max"] = max(overall[f]["max"], float(arr.max()))

        if (step - steps[0]) % STEP_INTERVAL == 0:
            print(f"\nStep {step}")
            for f in FIELDS:
                arr = np.array(data[f], dtype=np.float32).squeeze()
                mn, mx, mean, std = field_stats(arr)
                print(f"  {f:3s}: min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g}")

    print("\nOverall range across all steps:")
    for f in FIELDS:
        print(f"  {f:3s}: min={overall[f]['min']:.6g} max={overall[f]['max']:.6g}")


if __name__ == "__main__":
    main()