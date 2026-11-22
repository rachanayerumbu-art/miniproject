
import numpy as np

try:
    from security.defense import (
        assess_update,
        clip_update,
        defend_update,
        validate_update,
    )
except ModuleNotFoundError:
    from defense import (
        assess_update,
        clip_update,
        defend_update,
        validate_update,
    )


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


def test_normal_update_is_accepted():
    update = np.array([0.10, -0.05, 0.03])
    peer_updates = [
        np.array([0.08, -0.04, 0.02]),
        np.array([0.12, -0.06, 0.04]),
        np.array([0.09, -0.05, 0.03]),
    ]

    result = defend_update(
        update,
        client_id="client1",
        round_number=2,
        peer_updates=peer_updates,
        max_norm=0.5,
    )

    assert result["decision"] == "accepted"
    assert result["client_id"] == "client1"
    assert result["round_number"] == 2
    assert "Accepted" in result["reason"]
    print("PASS: Normal update is accepted")


def test_unusually_large_update_is_reduced():
    update = np.array([3.0, 4.0])

    result = defend_update(
        update,
        client_id="client3",
        round_number=1,
        max_norm=0.5,
    )

    assert result["decision"] == "reduced"
    assert np.isclose(
        np.linalg.norm(result["defended_update"]),
        0.5,
    )
    assert "suspicious-update evidence" in result["reason"]
    print("PASS: Unusually large update is reduced")


def test_suspicious_peer_update_is_flagged():
    peers = [
        np.array([0.10, -0.05, 0.02]),
        np.array([0.11, -0.04, 0.03]),
        np.array([0.09, -0.06, 0.02]),
    ]
    update = np.array([-3.0, 3.0, 3.0])

    result = defend_update(
        update,
        client_id="client5",
        round_number=3,
        peer_updates=peers,
        max_norm=1.0,
    )

    assert result["decision"] in ("flagged", "reduced", "rejected")
    assert "suspicious" in result["reason"].lower()
    print("PASS: Suspicious peer update is flagged or rejected")


def test_legitimate_distribution_variation_is_not_rejected():
    update = np.array([0.18, -0.12, 0.08])
    peers = [
        np.array([0.10, -0.08, 0.05]),
        np.array([0.12, -0.09, 0.06]),
        np.array([0.15, -0.10, 0.07]),
    ]

    result = defend_update(
        update,
        client_id="client4",
        round_number=4,
        peer_updates=peers,
        max_norm=1.0,
        suspicious_z_score=8.0,
    )

    assert result["decision"] == "accepted"
    assert "legitimately differ" in result["reason"]
    print("PASS: Legitimate data-distribution variation is preserved")


def test_empty_update_is_rejected():
    original = np.array([])

    result = defend_update(
        original,
        client_id="client2",
        round_number=1,
    )

    assert result["decision"] == "rejected"
    assert result["defended_update"] is None
    assert "empty" in result["reason"]
    print("PASS: Empty update is rejected")


def test_nan_update_is_rejected():
    original = np.array([0.1, np.nan, 0.2])

    result = defend_update(
        original,
        client_id="client2",
        round_number=2,
    )

    assert result["decision"] == "rejected"
    assert result["defended_update"] is None
    assert "NaN or infinite" in result["reason"]
    print("PASS: NaN update is rejected")


def test_infinite_update_is_rejected():
    original = np.array([0.1, np.inf, 0.2])

    result = defend_update(
        original,
        client_id="client2",
        round_number=2,
    )

    assert result["decision"] == "rejected"
    assert result["defended_update"] is None
    assert "NaN or infinite" in result["reason"]
    print("PASS: Infinite update is rejected")


def test_invalid_values_raise_in_validate_and_clip():
    for invalid_update in (
        np.array([]),
        np.array([np.nan]),
        np.array([np.inf]),
    ):
        try:
            validate_update(invalid_update)
            assert False, "validate_update should reject invalid values"
        except ValueError:
            pass

        try:
            clip_update(invalid_update)
            assert False, "clip_update should reject invalid values"
        except ValueError:
            pass

    print("PASS: Empty and non-finite values raise validation errors")


def test_original_update_is_not_modified():
    original = np.array([3.0, 4.0])
    before = original.copy()

    result = defend_update(
        original,
        client_id="client1",
        round_number=5,
        max_norm=0.5,
    )

    assert np.array_equal(original, before)
    assert not np.shares_memory(original, result["defended_update"])
    assert not np.shares_memory(original, result["original_update"])
    print("PASS: Original update is not modified")


def test_assess_alias_preserves_decision():
    update = np.array([0.1, 0.2])

    result = assess_update(
        update,
        client_id="client1",
        round_number=1,
        max_norm=0.5,
    )

    assert result["decision"] == "accepted"
    print("PASS: assess_update alias works")


if __name__ == "__main__":
    test_small_update_is_unchanged()
    test_large_update_is_clipped()
    test_update_direction_is_preserved()
    test_normal_update_is_accepted()
    test_unusually_large_update_is_reduced()
    test_suspicious_peer_update_is_flagged()
    test_legitimate_distribution_variation_is_not_rejected()
    test_empty_update_is_rejected()
    test_nan_update_is_rejected()
    test_infinite_update_is_rejected()
    test_invalid_values_raise_in_validate_and_clip()
    test_original_update_is_not_modified()
    test_assess_alias_preserves_decision()

    print("\nAll security defense tests passed!")
