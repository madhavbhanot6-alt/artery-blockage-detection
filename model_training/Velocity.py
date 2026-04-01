import numpy as np
import matplotlib.pyplot as plt



def generate_velocity_field(mask, vmax=1.0, asymmetry_strength=0.3):
    """
    Generates a velocity field inside an artery cross-section.

    Parameters:
        mask : 2D numpy array (1 inside artery, 0 outside)
        vmax : float → maximum center velocity
        asymmetry_strength : float → how distorted the flow is

    Returns:
        velocity : 2D numpy array of velocities
    """
    size = mask.shape[0]
    center = size // 2

    # Coordinate grid
    y, x = np.ogrid[:size, :size]

    # Distance from center
    r = np.sqrt((x - center)**2 + (y - center)**2)

    # Maximum radius of artery
    R = np.max(r * mask)

    # --- Laminar (Poiseuille) flow profile ---
    velocity = vmax * (1 - (r / R)**2)

    # Enforce zero velocity outside artery
    velocity *= mask

    # --- Introduce asymmetry (disturbed flow) ---
    shift_x = asymmetry_strength * R
    shift_y = asymmetry_strength * R

    r_shifted = np.sqrt((x - (center + shift_x))**2 +
                        (y - (center + shift_y))**2)

    disturbance = vmax * (1 - (r_shifted / R)**2)
    disturbance *= mask

    # Blend laminar flow with disturbance
    velocity = (1 - asymmetry_strength) * velocity + asymmetry_strength * disturbance

    # Remove negatives (physical safety)
    velocity = np.clip(velocity, 0, None)

    return velocity


if __name__ == "__main__":
    # Quick test using a dummy artery
    size = 128
    radius = 45

    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = ((x - center)**2 + (y - center)**2 <= radius**2).astype(np.float32)

    velocity = generate_velocity_field(mask)

    plt.imshow(velocity, cmap="inferno")
    plt.colorbar(label="Velocity")
    plt.title("Artery Velocity Field")
    plt.axis("off")
    plt.show()
