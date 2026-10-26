
import numpy as np


def create_model_update(size=5, seed=42):
    """Create a simulated, normal model update."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.1, size=size)


def create_malicious_update(normal_update, attack_strength=5.0):
    """Simulate a malicious client sending a harmful model update."""
    return -attack_strength * normal_update


if __name__ == "__main__":
    normal_update = create_model_update()
    malicious_update = create_malicious_update(normal_update)

    print("Normal client update:")
    print(normal_update)

    print("\nMalicious client update:")
    print(malicious_update)
