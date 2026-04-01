import numpy as np
import matplotlib.pyplot as plt
import random 
sz = 128
rd = random.randint(30,50)

def create_artery_mask(size=128, radius=50):
    """
    Creates a circular artery mask.
    
    Parameters:
        size   : int   → image size (size x size)
        radius : float → artery radius in pixels
    
    Returns:
        mask : 2D numpy array (1 inside artery, 0 outside)
    """
    center = size // 2

    # Create coordinate grid
    y, x = np.ogrid[:size, :size]

    # Distance from center
    distance_from_center = np.sqrt((x - center)**2 + (y - center)**2)

    # Circular mask
    mask = distance_from_center <= radius

    return mask.astype(np.float32)


if __name__ == "__main__":
    mask = create_artery_mask(size=sz, radius=rd)

    # Visual sanity check
    plt.imshow(mask, cmap="gray")
    plt.title("Artery Cross-Section Mask")
    plt.axis("off")
    plt.show()
