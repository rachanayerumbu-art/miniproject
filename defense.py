
import numpy as np


def clip_update(update, max_norm=0.5):
    """Limit the overall size of a model update."""
    update = np.asarray(update, dtype=float)

    norm = np.linalg.norm(update)

    if norm > max_norm:
        update = update * (max_norm / norm)

    return update


if __name__ == "__main__":
    malicious_update = np.array([
        -0.15235854,
        0.51999205,
        -0.37522560,
        -0.47028236,
        0.97551759
    ])

    defended_update = clip_update(malicious_update, max_norm=0.5)

    print("Original update norm:", np.linalg.norm(malicious_update))
    print("Defended update norm:", np.linalg.norm(defended_update))
    print("Defended update:")
    print(defended_update)
