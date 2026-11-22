try:
    from security.request_filter import predict_request
    from security.rate_limiter import check_rate_limit
except ModuleNotFoundError:
    from request_filter import predict_request
    from rate_limiter import check_rate_limit


def handle_request(record, source_id, time_source=None):
    """Handle one supplied IoT feature record."""
    classification = predict_request(record)
    prediction = classification.get("prediction")
    status = classification["status"]

    if status == "invalid":
        return {
            "status": "invalid",
            "reason": classification["reason"],
            "prediction": prediction,
        }

    if status == "blocked":
        return {
            "status": "blocked",
            "reason": classification["reason"],
            "prediction": prediction,
        }

    rate_result = check_rate_limit(source_id, time_source=time_source)

    if not rate_result["allowed"]:
        if rate_result["reason"].startswith("Invalid source_id"):
            return {
                "status": "invalid",
                "reason": rate_result["reason"],
                "prediction": prediction,
            }

        return {
            "status": "limited",
            "reason": rate_result["reason"],
            "prediction": prediction,
        }

    return {
        "status": "allowed",
        "reason": "Request classified as normal and within the rate limit",
        "prediction": prediction,
    }
