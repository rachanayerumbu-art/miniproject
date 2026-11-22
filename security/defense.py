
import numpy as np


def validate_update(update, name="update"):
    """Validate an update and return an independent NumPy copy."""
    if update is None:
        raise ValueError(f"{name} cannot be None")

    array = np.asarray(update, dtype=float)

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")

    return array.copy().reshape(-1)


def clip_update(update, max_norm=0.5):
    """Limit the overall size of a model update."""
    update = validate_update(update)

    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be a finite positive number")

    norm = float(np.linalg.norm(update))
    if norm > max_norm:
        return update * (max_norm / norm)

    return update


def _robust_peer_baseline(peer_updates):
    """Calculate norm and direction baselines without assuming equal peers."""
    valid_peers = []
    for peer_update in peer_updates:
        try:
            valid_peers.append(validate_update(peer_update, "peer update"))
        except (TypeError, ValueError):
            continue

    if not valid_peers:
        return None

    shapes = {peer.size for peer in valid_peers}
    if len(shapes) != 1:
        return None

    peer_matrix = np.vstack(valid_peers)
    peer_norms = np.linalg.norm(peer_matrix, axis=1)
    median_norm = float(np.median(peer_norms))
    norm_mad = float(np.median(np.abs(peer_norms - median_norm)))

    median_update = np.median(peer_matrix, axis=0)
    median_norm_value = float(np.linalg.norm(median_update))

    if median_norm_value == 0:
        median_direction = None
    else:
        median_direction = median_update / median_norm_value

    return {
        "median_norm": median_norm,
        "norm_mad": norm_mad,
        "median_update": median_update,
        "median_direction": median_direction,
    }


def defend_update(
    update,
    client_id=None,
    round_number=None,
    peer_updates=None,
    reference_update=None,
    max_norm=0.5,
    suspicious_z_score=3.5,
    reject_z_score=6.0,
):
    """
    Inspect a model parameter update and return a defensive decision.

    The result describes an evidence-based decision, not proof that a client
    is malicious. "reduced" means clipping was used; "rejected" is reserved
    for invalid updates or unusually severe norm/direction disagreement.
    """
    client_id = "unknown" if client_id is None else str(client_id)
    round_number = "unknown" if round_number is None else round_number
    base_result = {
        "decision": "rejected",
        "reason": "",
        "client_id": client_id,
        "round_number": round_number,
        "original_update": None,
        "defended_update": None,
        "original_norm": None,
    }

    try:
        original = validate_update(update)
    except (TypeError, ValueError) as error:
        base_result["reason"] = f"Rejected: invalid update ({error})."
        return base_result

    base_result["original_update"] = original
    original_norm = float(np.linalg.norm(original))
    base_result["original_norm"] = original_norm
    defended = original.copy()

    if not np.isfinite(max_norm) or max_norm <= 0:
        base_result["reason"] = (
            "Rejected: max_norm must be a finite positive number."
        )
        return base_result

    if suspicious_z_score <= 0 or reject_z_score <= 0:
        base_result["reason"] = (
            "Rejected: suspicious and rejection thresholds must be positive."
        )
        return base_result

    baseline = None
    peer_values = [] if peer_updates is None else peer_updates
    if peer_values:
        baseline = _robust_peer_baseline(peer_values)

    robust_z = None
    cosine_similarity = None
    opposite_reference = False

    if baseline is not None:
        median_norm = baseline["median_norm"]
        norm_mad = baseline["norm_mad"]
        robust_scale = 1.4826 * norm_mad
        if robust_scale > 0:
            robust_z = abs(original_norm - median_norm) / robust_scale

        direction = baseline["median_direction"]
        if direction is not None and original_norm > 0:
            cosine_similarity = float(
                np.dot(original, direction)
                / (np.linalg.norm(original) * np.linalg.norm(direction))
            )

    if reference_update is not None:
        try:
            reference = validate_update(reference_update, "reference update")
            if original.size == reference.size and original_norm > 0:
                reference_norm = float(np.linalg.norm(reference))
                if reference_norm > 0:
                    cosine_similarity = float(
                        np.dot(original, reference)
                        / (original_norm * reference_norm)
                    )
        except (TypeError, ValueError):
            pass

    if cosine_similarity is not None and cosine_similarity < -0.25:
        opposite_reference = True

    severely_suspicious = (
        robust_z is not None and robust_z >= reject_z_score
    ) or (
        original_norm > max_norm
        and opposite_reference
        and baseline is not None
        and robust_z is not None
        and robust_z >= suspicious_z_score
    )

    if severely_suspicious:
        defended = clip_update(original, max_norm=max_norm)
        base_result["decision"] = "rejected"
        base_result["defended_update"] = defended
        base_result["reason"] = (
            "Rejected/suspicious: update is an extreme norm outlier and conflicts "
            f"with peer direction (robust_norm_z={robust_z:.2f}). "
            "It was clipped before exclusion."
        )
        return base_result

    suspicious_norm = original_norm > max_norm
    suspicious_peer = (
        robust_z is not None and robust_z >= suspicious_z_score
    )
    suspicious_direction = opposite_reference

    if suspicious_norm:
        defended = clip_update(original, max_norm=max_norm)
        reasons = []
        reasons.append(
            f"L2 norm {original_norm:.6g} exceeds limit "
            f"{float(max_norm):.6g}"
        )
        if suspicious_peer:
            reasons.append(
                f"robust norm z-score is {robust_z:.2f}"
            )
        if suspicious_direction:
            reasons.append("update direction conflicts with peer/reference direction")

        base_result["decision"] = "reduced"
        base_result["defended_update"] = defended
        base_result["reason"] = (
            "Reduced/flagged: " + "; ".join(reasons)
            + ". This is suspicious-update evidence, not confirmation "
            "that the client is malicious."
        )
        return base_result

    if suspicious_peer or suspicious_direction:
        reasons = []
        if suspicious_peer:
            reasons.append(
                f"robust norm z-score is {robust_z:.2f}"
            )
        if suspicious_direction:
            reasons.append("update direction conflicts with peer/reference direction")

        base_result["decision"] = "flagged"
        base_result["defended_update"] = defended
        base_result["reason"] = (
            "Flagged: " + "; ".join(reasons)
            + ". The update is retained for this check because the evidence "
            "does not confirm malicious behavior."
        )
        return base_result

    base_result["decision"] = "accepted"
    base_result["defended_update"] = defended
    base_result["reason"] = (
        "Accepted: update norm and direction are consistent with the "
        "available baseline. Client data may legitimately differ."
    )
    return base_result


def assess_update(*args, **kwargs):
    """Alias for callers that prefer an inspection-oriented name."""
    return defend_update(*args, **kwargs)


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
