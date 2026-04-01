import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from Geometry import create_artery_mask
from Velocity import generate_velocity_field


# -------- Config --------
NUM_SAMPLES = 200
IMAGE_SIZE = 128
IMAGE_DIR = "data/processed/images"
LABEL_FILE = "data/processed/labels.csv"

os.makedirs(IMAGE_DIR, exist_ok=True)


# -------- Divergence / disturbance score --------
def compute_divergence_score(velocity, mask):
    """
    Simple disturbance metric based on velocity variation.
    Higher = more disturbed flow.
    """
    inside = velocity[mask == 1]

    mean_v = np.mean(inside)
    std_v = np.std(inside)

    # Coefficient of variation
    score = std_v / (mean_v + 1e-6)
    return float(score)


# -------- Dataset generation loop --------
with open(LABEL_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "divergence_score"])

    for i in range(NUM_SAMPLES):
        # Random parameters
        radius = random.randint(30, 50)
        asymmetry = random.uniform(0.0, 0.5)

        # Geometry
        mask = create_artery_mask(size=IMAGE_SIZE, radius=radius)

        # Velocity field
        velocity = generate_velocity_field(
            mask,
            vmax=1.0,
            asymmetry_strength=asymmetry
        )

        # Normalize velocity (0–1)
        v_min = velocity.min()
        v_max = velocity.max()
        velocity_norm = (velocity - v_min) / (v_max - v_min + 1e-6)

        # Save image
        image_name = f"slice_{i:04d}.png"
        image_path = os.path.join(IMAGE_DIR, image_name)

        plt.imsave(image_path, velocity_norm, cmap="gray")

        # Compute label
        score = compute_divergence_score(velocity, mask)

        # Write label
        writer.writerow([image_name, score])

print(f"Dataset created: {NUM_SAMPLES} samples")
print(f"Images saved to: {IMAGE_DIR}")
print(f"Labels saved to: {LABEL_FILE}")
