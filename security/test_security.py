
import numpy as np

from defense import clip_update


def test_small_update_is_unchanged():
    update = np.array([0.1, 0.2, 0.1])

    result = clip_update(update, max_norm=0.5)

    assert np.allclose(result, update)
    print("PASS: Small updates remain unchanged")


def test_large_update_is_clipped():
    update = np.array([3.0, 4.0])

    result = clip_update(update, max_norm=0.5)

    assert np.isclose(np.linalg.norm(result), 0.5)
    print("PASS: Large updates are clipped")


def test_update_direction_is_preserved():
    update = np.array([3.0, 4.0])

    result = clip_update(update, max_norm=0.5)

    assert np.allclose(result / np.linalg.norm(result),
                       update / np.linalg.norm(update))
    print("PASS: Update direction is preserved")


if __name__ == "__main__":
    test_small_update_is_unchanged()
    test_large_update_is_clipped()
    test_update_direction_is_preserved()

    print("\nAll security defense tests passed!")
