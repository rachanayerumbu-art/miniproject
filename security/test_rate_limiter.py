try:
    from security.rate_limiter import (
        REQUEST_LIMIT,
        check_rate_limit,
        reset_rate_limiter,
    )
except ModuleNotFoundError:
    from rate_limiter import (
        REQUEST_LIMIT,
        check_rate_limit,
        reset_rate_limiter,
    )


def test_ten_requests_allowed_and_eleventh_rejected():
    reset_rate_limiter()
    now = 1000.0

    for request_number in range(1, REQUEST_LIMIT + 1):
        result = check_rate_limit("device-1", lambda: now)
        assert result["allowed"] is True, result
        assert result["remaining"] == REQUEST_LIMIT - request_number, result

    blocked = check_rate_limit("device-1", lambda: now)
    assert blocked["allowed"] is False, blocked
    assert blocked["remaining"] == 0, blocked
    assert blocked["retry_after"] == 60.0, blocked
    print("PASS: The tenth request is allowed and the eleventh is rejected")


def test_requests_expire_after_sixty_seconds():
    reset_rate_limiter()

    for _ in range(REQUEST_LIMIT):
        result = check_rate_limit("device-expiry", lambda: 2000.0)
        assert result["allowed"] is True, result

    blocked = check_rate_limit("device-expiry", lambda: 2000.0)
    assert blocked["allowed"] is False, blocked

    allowed_again = check_rate_limit("device-expiry", lambda: 2060.0)
    assert allowed_again["allowed"] is True, allowed_again
    assert allowed_again["remaining"] == 9, allowed_again
    print("PASS: Requests expire after 60 seconds")


def test_devices_have_separate_windows():
    reset_rate_limiter()
    now = 3000.0

    for _ in range(REQUEST_LIMIT):
        assert check_rate_limit("device-a", lambda: now)["allowed"] is True

    assert check_rate_limit("device-a", lambda: now)["allowed"] is False

    device_b = check_rate_limit("device-b", lambda: now)
    assert device_b["allowed"] is True, device_b
    assert device_b["remaining"] == 9, device_b
    print("PASS: Separate devices have separate rate-limit windows")


def test_invalid_source_ids_are_rejected():
    reset_rate_limiter()
    invalid_source_ids = (
        None,
        "",
        "   ",
        0,
        -1,
        1.5,
        True,
        [],
        {},
    )

    for source_id in invalid_source_ids:
        result = check_rate_limit(source_id, lambda: 4000.0)
        assert result["allowed"] is False, (source_id, result)
        assert result["remaining"] == 0, (source_id, result)
        assert "Invalid source_id" in result["reason"], (source_id, result)

    print("PASS: Invalid source IDs are rejected")


def test_invalid_time_sources_are_rejected():
    reset_rate_limiter()

    invalid_times = (None, True, "4000", -1.0)
    for invalid_time in invalid_times:
        result = check_rate_limit("device-time", lambda: invalid_time)
        assert result["allowed"] is False, (invalid_time, result)
        assert result["remaining"] == 0, (invalid_time, result)

    print("PASS: Invalid time sources are rejected")


def test_positive_integer_source_id_is_normalized():
    reset_rate_limiter()

    result_one = check_rate_limit(123, lambda: 5000.0)
    result_two = check_rate_limit("123", lambda: 5000.0)

    assert result_one["allowed"] is True, result_one
    assert result_two["allowed"] is True, result_two
    assert result_two["remaining"] == 8, result_two
    print("PASS: Positive integer source IDs are normalized consistently")


if __name__ == "__main__":
    test_ten_requests_allowed_and_eleventh_rejected()
    test_requests_expire_after_sixty_seconds()
    test_devices_have_separate_windows()
    test_invalid_source_ids_are_rejected()
    test_invalid_time_sources_are_rejected()
    test_positive_integer_source_id_is_normalized()
    reset_rate_limiter()
    print("\nAll rate limiter tests passed!")
